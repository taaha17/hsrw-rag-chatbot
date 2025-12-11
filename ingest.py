"""
Document Ingestion and Parsing Pipeline

This script processes PDF documents from Rhine-Waal University:
1. ISE_MH.pdf (Module Handbook) → Extract module metadata
2. ISE_CS.pdf (Class Schedule) → Parse structured schedule data
3. ISE_ER_*.pdf (Examination Regulations) → General text chunking

Key Challenges Solved:
- Multi-line module names in PDFs (e.g., "Physics: Mechanics, Electricity and\nMagnetism")
- Split professor names (e.g., "Prof. Dr. Große-\nKampmann" on two lines)
- Distinguishing table headers from actual module entries
- Block course dates and building codes mixed with schedule data

Output:
- module_map.json: Mapping of module codes to names (e.g., CI_3.02 → "Signals and systems")
- class_schedule.json: Structured schedule with day, time, professor, room
- chroma_db/: Vector database with embedded document chunks

Usage:
    python ingest.py
    
This should be run whenever PDFs are updated.
"""

import os
import shutil
import json
import re
import requests
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils import HttpOllamaEmbeddings
from config import (
    DATA_FOLDER,
    DB_PATH,
    MODULE_MAP_PATH,
    CLASS_SCHEDULE_PATH,
    EMBEDDING_MODEL,
    MODULE_PATTERN,
)


# Compile the regex pattern for efficiency
MODULE_PATTERN_RE = re.compile(MODULE_PATTERN)

def is_valid_header(line, name):
    """
    A set of heuristic rules to determine if a line is a valid module title.
    This is highly specific to the format of the HSRW module handbook PDF.
    """
    name = name.strip()
    
    # 1. Reject Table of Contents lines (End with dots and a number)
    # Example: "Fundamentals of CS ....... 4"
    if re.search(r"\.+\s*\d+$", line.strip()):
        return False

    # 2. Reject metadata keywords (Workload, CP, etc.)
    bad_keywords = ["150 h", "300 h", "CP", "semester", "Workload", "Duration", "Code"]
    for word in bad_keywords:
        if word in name:
            return False
            
    # 3. Reject raw table stats (e.g., "4 C 5")
    if re.search(r'^\d\s+[A-Z]\s+\d', name):
        return False
        
    # 4. Reject description starts (lowercase or quotes)
    if name[0].islower() or name.startswith('"') or name.startswith('“') or name.startswith('’'):
        return False
        
    # 5. Reject too short names
    if len(name) < 5:
        return False

    # 6. Reject names ending with digits (common in tables, e.g. "Data Science 5")
    if re.search(r'\d+$', name):
        return False

    # 7. Reject names with table-like keywords
    if any(x in name for x in ["ECTS", "SWS", "Exam", "graded", "written"]):
        return False
        
    return True

def parse_module_handbook(file_path, filename):
    """
    Parses a PDF module handbook to extract modules and their content.
    This is not a "semantic" chunker, but a layout-aware parser tailored
    to the specific format of the HSRW PDF.
    """
    loader = PDFPlumberLoader(file_path)
    raw_docs = loader.load()
    
    # Pre-cleaning: Remove header/footer noise if possible, but raw read is safer
    full_text = "\n".join([doc.page_content for doc in raw_docs])
    lines = full_text.split('\n')
    
    chunks = []
    module_map = {}
    
    current_code = None
    current_title = None
    current_buffer = []
    
    print(f"   - Parsing {filename}...")

    for line in lines:
        line = line.strip()
        if not line: continue

        match = MODULE_PATTERN_RE.match(line)
        
        if match:
            code = match.group(1)
            raw_name = match.group(2).strip()
            
            # CRITICAL CHECK: Use the full line to detect TOC dots
            if is_valid_header(line, raw_name):
                
                # Save previous module
                if current_code:
                    chunks.append(Document(
                        page_content="\n".join(current_buffer),
                        metadata={"source": filename, "code": current_code, "title": current_title}
                    ))
                
                # Start new module
                current_code = code
                current_title = raw_name
                current_buffer = [line]
                
                # Only add to map if not already there (prioritizes the main definition over TOC)
                if current_code not in module_map:
                    module_map[current_code] = current_title
                    print(f"     ✅ Found: {current_code} - {current_title}")
            else:
                # It matched regex but was a TOC line or table row -> treat as content
                if current_code:
                    current_buffer.append(line)
        else:
            # Regular text
            if current_code:
                current_buffer.append(line)

    # Save the very last module
    if current_code:
        chunks.append(Document(
            page_content="\n".join(current_buffer),
            metadata={"source": filename, "code": current_code, "title": current_title}
        ))

    return chunks, module_map

def parse_class_schedule(file_path, filename):
    """
    Parses a class schedule PDF to extract structured data about class sessions.
    
    This parser uses a state machine approach to handle:
    - Multi-line module names (e.g., "Physics: Mechanics, Electricity and\nMagnetism")
    - Multi-word professor names (e.g., "Prof. Dr. Ressel", "Ms. Yacizi")
    - Various room types and codes
    - Block course information (treated as notes, not parsed separately)
    
    The PDF structure is:
    Time Time Code ModuleName Type Professor Room [Building/Code]
    12:15 15:30 8313 Physics: Mechanics... L&E Prof. Dr. Ressel Hörsaal 1 01 00 215
    """
    print(f"   - Parsing schedule from {filename}...")
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    lines = full_text.split('\n')

    schedule = []
    current_day = None
    current_semester = None
    
    # Track if we're continuing a multi-line module name
    pending_entry = None
    pending_block_dates = []  # Accumulate block dates for current entry

    # Regex patterns for identifying lines
    semester_pattern = re.compile(r'(\d)\.\s*Semester')
    day_pattern = re.compile(r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Montag|Dienstag|Mittwoch|Donnerstag|Freitag)')
    
    # Main entry pattern: Start time, end time, module code (4 digits)
    entry_start_pattern = re.compile(r'^(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+(\d{4})\s+(.+)$')
    
    # Pattern for block course dates
    block_date_pattern = re.compile(r'Block course:\s*(.+)')
    
    # Pattern to match building/floor/room codes (e.g., "01 01 110" or "7 03 130")
    # format: building (1-2 digits) + floor (2 digits) + room (3 digits)
    room_code_pattern = re.compile(r'\b(\d{1,2})\s+(\d{2})\s+(\d{3})\b')
    
    # Room types we're looking for
    room_keywords = ['Hörsaal', 'Seminarraum', 'Labor', 'RAG', 'Cloud Resilience Lab', 
                     'IOT Lab', 'E-Technik Labor', 'd i g i t a l / o n l i n e', 'tba']
    
    # Class type codes (appear before professor name)
    type_codes = ['L&E', 'L', 'E', 'P', 'PT', 'SL']
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and footer lines
        if not line or 'SEITE' in line or 'VON' in line:
            continue

        # Check for semester marker
        semester_match = semester_pattern.search(line)
        if semester_match:
            # Save any pending entry before switching semester
            if pending_entry:
                if pending_block_dates:
                    pending_entry["block_dates"] = ", ".join(pending_block_dates)
                schedule.append(pending_entry)
            
            current_semester = int(semester_match.group(1))
            pending_entry = None  # Reset any pending entry
            pending_block_dates = []  # Reset block dates
            continue

        # Check for day marker
        day_match = day_pattern.match(line)
        if day_match:
            # Save any pending entry before switching day
            if pending_entry:
                if pending_block_dates:
                    pending_entry["block_dates"] = ", ".join(pending_block_dates)
                schedule.append(pending_entry)

            current_day = day_match.group(1)
            # Normalize German day names to English
            day_mapping = {
                'Montag': 'Monday', 'Dienstag': 'Tuesday', 'Mittwoch': 'Wednesday',
                'Donnerstag': 'Thursday', 'Freitag': 'Friday'
            }
            current_day = day_mapping.get(current_day, current_day)
            pending_entry = None  # Reset any pending entry
            pending_block_dates = []  # Reset block dates
            continue

        # Check if this is the start of a new schedule entry
        entry_match = entry_start_pattern.match(line)
        
        if entry_match:
            # Save any pending entry from previous iteration (with its block dates)
            if pending_entry:
                if pending_block_dates:
                    pending_entry["block_dates"] = ", ".join(pending_block_dates)
                schedule.append(pending_entry)
                pending_entry = None
                # IMPORTANT: do NOT reset block dates here - they may apply to next entry too
            
            start_time = entry_match.group(1)
            end_time = entry_match.group(2)
            module_code = entry_match.group(3)
            rest_of_line = entry_match.group(4)
            
            # reset block dates for each new entry
            pending_block_dates = []
            
            # Now parse the rest: ModuleName Type Professor Room [Building Floor RoomNum]
            module_name = ""
            class_type = ""
            professor = ""
            room = ""
            building_code = None
            floor_code = None
            room_code = None
            
            # Find the class type in the line
            type_found = False
            for t_code in type_codes:
                if f' {t_code} ' in rest_of_line:
                    parts = rest_of_line.split(f' {t_code} ', 1)
                    module_name = parts[0].strip()
                    class_type = t_code
                    after_type = parts[1] if len(parts) > 1 else ""
                    type_found = True
                    
                    # Extract room codes if present (e.g., "01 01 110" or "7 03 130")
                    room_code_match = room_code_pattern.search(after_type)
                    if room_code_match:
                        building_code = room_code_match.group(1)  # building number
                        floor_code = room_code_match.group(2)     # floor number
                        room_code = room_code_match.group(3)      # room number (3 digits)
                        # Remove the codes from after_type for easier processing
                        after_type = room_code_pattern.sub('', after_type).strip()
                    
                    # Now find the room in the after_type section
                    room_found = False
                    for room_kw in room_keywords:
                        if room_kw in after_type:
                            # Split by the keyword
                            split_parts = after_type.split(room_kw, 1)
                            professor = split_parts[0].strip()
                            
                            # Capture the room name including any number after it (e.g. "Hörsaal 3")
                            room_suffix = split_parts[1].strip()
                            room = f"{room_kw} {room_suffix}".strip()
                            
                            room_found = True
                            break
                    
                    if not room_found:
                        # No room keyword found, treat everything as professor
                        professor = after_type.strip()
                        room = "tba"
                    
                    break
            
            if type_found:
                # We successfully parsed this entry
                pending_entry = {
                    "semester": current_semester,
                    "day": current_day,
                    "start_time": start_time,
                    "end_time": end_time,
                    "module_code": module_code,
                    "module_name": module_name,
                    "type": class_type,
                    "professor": professor,
                    "room": room,
                    "building": building_code,
                    "floor": floor_code,
                    "room_number": room_code
                }
            else:
                # Type code not found on this line - skip
                print(f"⚠️ WARNING: Matched entry pattern but no type found: {line}")
                pass
        
        elif pending_entry:
            # This is a continuation line - could be:
            # 1. Block course dates (e.g., "Block course: 29.09.25, 03.11.25, ...")
            # 2. Rest of professor name (e.g., "Kampmann" after "Prof. Dr. Große-")
            # 3. Module name continuation
            # 4. Metadata (skip)
            
            # Check for block course dates
            block_match = block_date_pattern.search(line)
            if block_match:
                # Extract all dates from this line and potential continuation lines
                dates_str = block_match.group(1).strip()
                # Clean up and store (remove trailing commas)
                if dates_str and not dates_str.startswith("("):
                    # Remove trailing comma if present
                    dates_str = dates_str.rstrip(',')
                    pending_block_dates.append(dates_str)
                continue
            
            # Check if it's a continuation of block dates (dates without "Block course:" prefix)
            if pending_block_dates and re.match(r'[\d.,\s]+', line) and '.' in line:
                # Clean trailing comma
                clean_line = line.strip().rstrip(',')
                pending_block_dates.append(clean_line)
                continue
            
            # Skip other metadata lines
            if any(skip_word in line for skip_word in ["Start:", "biweekly", "Gruppe", "weekly", "SEITE", "VON"]):
                continue
            
            # Check if this looks like a name continuation (for professor)
            # Professor names ending with "-" usually continue on next line
            if pending_entry["professor"].endswith("-"):
                # This line is likely the rest of the professor's name
                pending_entry["professor"] += line.strip()
                continue
            
            # Check if this looks like building/room codes (numbers with spaces)
            # Example: "01 01 110" or "7 03 130" (building floor room)
            if re.match(r'^\d{1,2}\s+\d{2}\s+\d{3}$', line.strip()):
                # Extract and store room codes
                room_code_match = room_code_pattern.match(line.strip())
                if room_code_match:
                    pending_entry["building"] = room_code_match.group(1)
                    pending_entry["floor"] = room_code_match.group(2)
                    pending_entry["room_number"] = room_code_match.group(3)  # fixed: was group(4), should be group(3)
                continue
            
            # If it's a single word and the professor field is short, might be completing the professor name
            if len(line.split()) == 1 and len(pending_entry["professor"]) < 30:
                # Could be part of professor name or location
                if not ("Prof." in pending_entry["professor"] or "Mr." in pending_entry["professor"] or "Ms." in pending_entry["professor"]):
                    pending_entry["professor"] += " " + line.strip()
                continue
            
            # Otherwise, treat as continuation of module name
            # Module names can be split across lines (e.g., "Physics: Mechanics, Electricity and\nMagnetism")
            if len(line) > 2 and not line.startswith(("Prof.", "Mr.", "Ms.", "Dr.", "Block")):
                # Don't append if it's just numbers (building codes)
                if not re.match(r'^\d+\s+\d{2}', line):
                    pending_entry["module_name"] += " " + line.strip()
    
    # Don't forget to save the last pending entry with its block dates
    if pending_entry:
        if pending_block_dates:
            pending_entry["block_dates"] = ", ".join(pending_block_dates)
        schedule.append(pending_entry)

    return schedule

def validate_schedule_data(schedule_data, module_map):
    """
    validate extracted schedule data for completeness and quality.
    logs warnings for potential issues.
    
    returns: dict with validation statistics
    """
    stats = {
        'total_entries': len(schedule_data),
        'incomplete_entries': 0,
        'missing_building_codes': 0,
        'modules_with_schedule': set(),
        'modules_without_schedule': []
    }
    
    print("\\n🔍 validating extracted data...")
    
    # check each schedule entry for completeness
    for entry in schedule_data:
        has_issue = False
        
        # check for missing building codes
        if not entry.get('building') or not entry.get('room_number'):
            stats['missing_building_codes'] += 1
            has_issue = True
        
        # check for incomplete data
        required_fields = ['semester', 'day', 'start_time', 'end_time', 'module_code', 'module_name', 'professor']
        missing_fields = [f for f in required_fields if not entry.get(f)]
        if missing_fields:
            stats['incomplete_entries'] += 1
            print(f"  ⚠️  incomplete entry for {entry.get('module_name', 'unknown')}: missing {', '.join(missing_fields)}")
            has_issue = True
        
        # track which modules have schedules
        if entry.get('module_code'):
            stats['modules_with_schedule'].add(entry['module_code'])
    
    # check for modules in map without schedule entries (electives are expected to have no schedule)
    for code in module_map.keys():
        # extract semester identifier (CI_3.02 → 3, CI_W.01 → W)
        parts = code.split('_')
        if len(parts) >= 2:
            sem_id = parts[1].split('.')[0]
            # skip electives (W) and key competences (K) - they don't appear in schedule
            if sem_id.isdigit():
                # this is a regular semester module, should have schedule
                # but we don't have numeric schedule codes in our current data
                # so we'll just track what we have
                pass
    
    # print summary
    print(f"\\n📊 validation summary:")
    print(f"  ✅ total schedule entries: {stats['total_entries']}")
    print(f"  ⚠️  entries missing building codes: {stats['missing_building_codes']}")
    print(f"  ⚠️  incomplete entries: {stats['incomplete_entries']}")
    print(f"  📚 unique module codes in schedule: {len(stats['modules_with_schedule'])}")
    
    if stats['incomplete_entries'] == 0 and stats['missing_building_codes'] == 0:
        print("  🎉 all data looks good!")
    
    return stats


def ingest_documents():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"🧹 Cleared database.")

    all_chunks = []
    master_module_map = {}
    schedule_data = []

    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder '{DATA_FOLDER}' not found.")
        return

    for filename in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if filename.endswith(".pdf") and "MH" in filename:
            chunks, mod_map = parse_module_handbook(file_path, filename)
            all_chunks.extend(chunks)
            master_module_map.update(mod_map)
        elif filename.endswith(".pdf") and "CS" in filename:
            schedule_data = parse_class_schedule(file_path, filename)
        elif filename.endswith(".pdf"):
            print(f"   - Standard parsing {filename}...")
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            for doc in split_docs:
                doc.metadata["source"] = filename
            all_chunks.extend(split_docs)

    with open(MODULE_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(master_module_map, f, indent=4)
    print(f"✅ Saved {len(master_module_map)} clean modules to map.")

    with open(CLASS_SCHEDULE_PATH, "w", encoding="utf-8") as f:
        json.dump(schedule_data, f, indent=4)
    print(f"✅ Saved {len(schedule_data)} schedule entries.")
    
    # validate the extracted data
    if schedule_data and master_module_map:
        validate_schedule_data(schedule_data, master_module_map)

    if all_chunks:
        print(f"🧠 Embedding {len(all_chunks)} chunks...")
        embedding_model = HttpOllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )
        print(f"✅ Database created.")

if __name__ == "__main__":
    ingest_documents()