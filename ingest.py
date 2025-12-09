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

    # Regex patterns for identifying lines
    semester_pattern = re.compile(r'(\d)\.\s*Semester')
    day_pattern = re.compile(r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Montag|Dienstag|Mittwoch|Donnerstag|Freitag)')
    
    # Main entry pattern: Start time, end time, module code (4 digits)
    # We'll capture everything after the code and parse it more carefully
    entry_start_pattern = re.compile(r'^(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+(\d{4})\s+(.+)$')
    
    # Room types we're looking for - these help us know when the module name ends
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
            current_semester = int(semester_match.group(1))
            pending_entry = None  # Reset any pending entry
            continue

        # Check for day marker
        day_match = day_pattern.match(line)
        if day_match:
            current_day = day_match.group(1)
            # Normalize German day names to English
            day_mapping = {
                'Montag': 'Monday', 'Dienstag': 'Tuesday', 'Mittwoch': 'Wednesday',
                'Donnerstag': 'Thursday', 'Freitag': 'Friday'
            }
            current_day = day_mapping.get(current_day, current_day)
            pending_entry = None  # Reset any pending entry
            continue

        # Check if this is the start of a new schedule entry
        entry_match = entry_start_pattern.match(line)
        
        if entry_match:
            # Save any pending entry from previous iteration
            if pending_entry:
                schedule.append(pending_entry)
                pending_entry = None
            
            start_time = entry_match.group(1)
            end_time = entry_match.group(2)
            module_code = entry_match.group(3)
            rest_of_line = entry_match.group(4)
            
            # Now parse the rest: ModuleName Type Professor Room [codes...]
            # Strategy: Find the type code, then work backwards for module name
            # and forwards for professor and room
            
            module_name = ""
            class_type = ""
            professor = ""
            room = ""
            
            # Find the class type in the line
            type_found = False
            for t_code in type_codes:
                if f' {t_code} ' in rest_of_line:
                    parts = rest_of_line.split(f' {t_code} ', 1)
                    module_name = parts[0].strip()
                    class_type = t_code
                    after_type = parts[1] if len(parts) > 1 else ""
                    type_found = True
                    
                    # Now find the room in the after_type section
                    room_found = False
                    for room_kw in room_keywords:
                        if room_kw in after_type:
                            before_room = after_type.split(room_kw)[0].strip()
                            professor = before_room
                            room = room_kw
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
                    "room": room
                }
            else:
                # Type code not found on this line - might be a formatting issue
                # or the line is incomplete. Skip for now.
                pass
        
        elif pending_entry:
            # This is a continuation line - could be:
            # 1. Rest of professor name (e.g., "Kampmann" after "Prof. Dr. Große-")
            # 2. Module name continuation
            # 3. Block course info (skip these)
            
            # Skip block course notes and start dates
            if any(skip_word in line for skip_word in ["Block course:", "Start:", "biweekly", "Gruppe", "weekly"]):
                continue
            
            # Check if this looks like a name continuation (for professor)
            # Professor names ending with "-" usually continue on next line
            if pending_entry["professor"].endswith("-"):
                # This line is likely the rest of the professor's name
                pending_entry["professor"] += line.strip()
                continue
            
            # Check if this looks like building/room codes (numbers with spaces)
            # Example: "1 01 00 215" or "03 02 110"
            if re.match(r'^\d+\s+\d{2}\s+\d{2}\s+\d{3}$', line.strip()):
                # This is building/room code, skip it
                continue
            
            # If it's a single word and the professor field is short, it might be completing the professor name
            if len(line.split()) == 1 and len(pending_entry["professor"]) < 30:
                # Could be part of professor name or location
                # If professor already looks complete, skip this (it's probably a building name)
                if not ("Prof." in pending_entry["professor"] or "Mr." in pending_entry["professor"] or "Ms." in pending_entry["professor"]):
                    pending_entry["professor"] += " " + line.strip()
                # Otherwise skip (it's location info)
                continue
            
            # Otherwise, treat as continuation of module name (rare but possible)
            # Only append if it doesn't look like metadata
            if len(line) > 5 and not line.startswith(("Prof.", "Mr.", "Ms.", "Dr.")):
                pending_entry["module_name"] += " " + line.strip()
    
    # Don't forget to save the last pending entry
    if pending_entry:
        schedule.append(pending_entry)

    return schedule

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