# Chatbot Improvements - Implementation Guide

## Overview
This document explains all the improvements made to the HSRW RAG Chatbot to fix accuracy issues with class schedule queries and improve overall query understanding.

## Problems Identified

### 1. **Inaccurate Schedule Parsing**
- **Issue**: Module names were truncated (e.g., "Physics: Mechanics, Electricity and" instead of full name)
- **Root Cause**: Single-line regex couldn't handle multi-line module names in PDF
- **Impact**: 35 schedule entries with typos and incomplete data

### 2. **Poor Schedule Query Handling**
- **Issue**: Chatbot said "I don't have access" even when schedule data existed
- **Root Cause**: Chat logic didn't properly prioritize structured schedule data over RAG
- **Impact**: Users got generic "contact advisor" responses instead of exact schedules

### 3. **Weak Query Understanding**
- **Issue**: Couldn't detect different query types (schedule vs. module info vs. lists)
- **Root Cause**: Simple keyword matching instead of intent classification
- **Impact**: Wrong data sources used for queries

## Solutions Implemented

### 1. Enhanced PDF Parser (`ingest.py`)

**New State Machine Parser**:
```python
def parse_class_schedule(file_path, filename):
    """
    Uses state machine to handle:
    - Multi-line module names
    - Split professor names (e.g., "Prof. Dr. Große-\nKampmann")
    - Block course notes (ignored)
    - Building codes (filtered out)
    """
```

**Key Improvements**:
- Tracks `pending_entry` to accumulate multi-line data
- Detects professor name continuations (names ending with `-`)
- Filters out building codes and metadata lines
- Result: Clean, complete schedule data (21 entries, down from 35 incorrect ones)

### 2. Smart Query Routing (`chat.py`)

**New Intent-Based Routing**:
```python
# Step 1: Detect what user wants
intent = detect_query_intent(query)

# Step 2: Route to appropriate data source
if intent == 'schedule':
    # Use structured schedule_data
elif intent == 'modules_list':
    # Use module_map
elif intent == 'module_info':
    # Use RAG retriever
else:
    # General RAG
```

**Key Benefits**:
- Schedule queries → structured JSON (no hallucinations)
- Module lists → exact module_map data
- Detailed questions → RAG with context
- No more "I don't have access" for data we have!

### 3. Intent Detection System (`logic_engine.py`)

**New Function: `detect_query_intent()`**:
```python
def detect_query_intent(query):
    """
    Analyzes query to determine:
    - intent: 'schedule' | 'modules_list' | 'module_info' | 'general'
    - Additional context about what's needed
    """
```

**Supported Intent Types**:
1. **Schedule**: when, time, day, class, today, tomorrow, where, room, professor
2. **Modules List**: what modules, my courses, curriculum, list of
3. **Module Info**: tell me about, who teaches, credits, prerequisites
4. **General**: everything else (use RAG)

### 4. Improved Module Matching

**Enhanced `find_code_by_name()`**:
- **Scoring system**: 
  - Exact substring match: 100 points
  - All words match: 50+ points
  - Partial match: 10 points per word
- **Better stopword filtering**: Removes "when", "where", "my", etc.
- **Handles variations**: "signals and systems" finds "Signals and Systems"

**Example**:
```
Query: "when is signals and systems class?"
→ Removes: "when", "is", "class"
→ Searches for: "signals", "and", "systems"
→ Finds: CI_3.02 - "Signals and systems"
→ Returns schedule data
```

### 5. Better LLM Prompting

**Visual Structure**:
```python
system_prompt = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OFFICIAL CLASS SCHEDULE DATA]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schedule_data}

YOUR TASK:
1. READ the schedule data carefully
2. Extract the relevant information
3. Present it CLEARLY to the student
4. DO NOT say "I don't have access"
"""
```

**Key Strategies**:
- Clear visual separation of data sections
- Explicit instructions (DO/DON'T format)
- Examples of good vs. bad responses
- Different prompts for schedule vs. general queries

## Testing Results

### Test Script Output (`test_queries.py`)

```
Query: "When is signals and systems class?"
→ Intent: schedule
→ Module Found: CI_3.02
→ Schedule: Monday 14:00-15:30, Prof. Dr. Strumpen, Hörsaal

Query: "What classes do I have today? I'm in 3rd semester"
→ Intent: schedule
→ Semester: 3
→ Found 3 classes:
   - 10:00-11:30: Higher Mathematics
   - 12:15-13:45: Data Management
   - 14:00-15:30: Software engineering Group 2

Query: "What modules do I have in 2nd semester?"
→ Intent: modules_list
→ Semester: 2
→ 5 modules found (correct list shown)
```

### Data Quality Improvements

**Before**:
```json
{
  "module_name": "Physics: Mechanics, Electricity and",
  "professor": "Prof. Dr. Ressel",
  "room": "Hörsaal"
}
```

**After**:
```json
{
  "module_name": "Physics: Mechanics, Electricity and Magnetism",
  "professor": "Prof. Dr. Große-Kampmann",
  "room": "Hörsaal"
}
```

## File Changes Summary

### Modified Files:

1. **`ingest.py`** (150+ lines changed)
   - Rewrote `parse_class_schedule()` with state machine
   - Added multi-line handling logic
   - Improved name continuation detection

2. **`logic_engine.py`** (100+ lines added)
   - Added `detect_query_intent()` function
   - Enhanced `find_code_by_name()` with scoring
   - Added `get_all_schedule_for_semester()` helper
   - Improved docstrings and type hints

3. **`chat.py`** (200+ lines changed)
   - Complete rewrite of query routing logic
   - Added intent-based data source selection
   - Improved LLM prompt structure
   - Better context building

4. **`config.py`** (15 lines added)
   - Added `DAY_NAMES` constant
   - Added `CLASS_TYPES` dictionary
   - Better organization

5. **`README.md`** (100+ lines)
   - Complete documentation rewrite
   - Architecture diagram
   - Usage examples
   - Development guide

### New Files:

1. **`test_queries.py`** (New)
   - Test script for validating query logic
   - No need for full chatbot to test
   - Shows intent detection and data retrieval

## Architecture Changes

### Before:
```
User Query → Simple keyword check → RAG or Module Map → LLM → Response
```

### After:
```
User Query 
  → detect_query_intent()
    → Route to appropriate source:
       ├─ Schedule Query → schedule_data (JSON)
       ├─ List Query → module_map (JSON)
       ├─ Info Query → RAG retriever + schedule_data
       └─ General → RAG retriever
  → Structured context building
  → Intent-specific LLM prompt
  → Response
```

## Key Design Decisions

### 1. **Prioritize Structured Data**
**Rationale**: For factual queries (schedules, dates, names), structured JSON is more reliable than RAG retrieval. LLMs can hallucinate, but JSON can't.

**Implementation**: Check intent first, use schedule_data for schedule queries before falling back to RAG.

### 2. **State Machine Parser**
**Rationale**: PDFs have complex multi-line layouts that simple regex can't handle.

**Implementation**: Track pending_entry state, accumulate lines until next entry starts.

### 3. **Explicit LLM Instructions**
**Rationale**: Generic prompts lead to "I don't have access" responses even when data is provided.

**Implementation**: Use visual separators, mandatory instructions, and explicit DO/DON'T lists.

### 4. **Fuzzy Module Matching**
**Rationale**: Users say "physics" not "Physics: Mechanics, electricity and magnetism"

**Implementation**: Scoring system that rewards substring matches and word overlaps.

## Future Improvements

### Recommended Enhancements:

1. **Semester Context Persistence**
   - Store user's semester in conversation history
   - Auto-fill semester for follow-up queries
   
2. **Time-Aware Queries**
   - "What's my next class?" → Check current time
   - "Classes tomorrow" → Use date arithmetic

3. **Multi-Module Queries**
   - "When are my labs?" → Filter by type='P'
   - "Show all Thursday classes" → Full day view

4. **Web Interface**
   - Replace terminal with Streamlit/Gradio UI
   - Add calendar view for schedules
   - Export to ICS calendar format

5. **Additional Data Sources**
   - Exam schedules
   - Office hours
   - Room availability
   - Course registration deadlines

## Maintenance Guide

### When PDFs Update:

1. Replace PDF in `data/` folder
2. Run `python ingest.py`
3. Verify `class_schedule.json` looks correct
4. Test with `python test_queries.py`
5. Launch chatbot with `python chat.py`

### Adding New Semesters:

Update `config.py`:
```python
WINTER_SEMS = [1, 3, 5, 7, 9]  # If program extends to 9 sems
```

### Changing LLM Model:

Update `config.py`:
```python
CHAT_MODEL = "llama3.3"  # or "mistral", "phi3", etc.
```

### Debugging Query Issues:

1. Check intent detection: `python test_queries.py`
2. Verify module matching: Look at test script output
3. Check schedule data: Open `data/class_schedule.json`
4. Test prompt: Read the system_prompt in chat.py

## Common Issues & Solutions

### Issue: "Module not found"
**Solution**: Check module_map.json, verify exact spelling, adjust find_code_by_name() scoring

### Issue: Schedule data missing
**Solution**: Re-run ingest.py, check PDF structure matches expected format

### Issue: LLM gives generic responses
**Solution**: Review system prompt, ensure structured data is clearly marked with visual separators

### Issue: Professor names incomplete
**Solution**: Check parse_class_schedule() continuation logic, verify hyphen detection

## Performance Metrics

- **Ingestion time**: ~10-15 seconds for all PDFs
- **Schedule entries**: 21 (clean, complete data)
- **Module map**: 37 modules
- **Vector chunks**: 67 embedded documents
- **Query response time**: 2-5 seconds (depends on Ollama)

## Conclusion

The chatbot now provides accurate, helpful responses to schedule and module queries by:
1. ✅ Parsing PDFs correctly (complete names, no typos)
2. ✅ Understanding query intent (schedule vs. info vs. list)
3. ✅ Prioritizing structured data (no hallucinations)
4. ✅ Using clear LLM prompts (explicit instructions)

Students can now get immediate, accurate answers to common questions without contacting advisors for basic information.

---
**Last Updated**: December 9, 2025
**Version**: 2.0 (Major refactor)
