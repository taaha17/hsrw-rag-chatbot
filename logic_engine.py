"""
Logic Engine - Query Understanding and Data Filtering

This module contains all the "intelligence" for understanding user queries:
- Intent detection: What is the user asking for?
- Semester extraction: Which semester is relevant?
- Module matching: Find modules by fuzzy name search
- Data filtering: Get the right subset of data

Think of this as the "brain" that interprets natural language queries
and translates them into structured data requests.

Example Flow:
1. User asks: "when is my physics lecture?"
2. detect_query_intent() → 'schedule'
3. find_code_by_name() → 'CI_1.07'
4. get_schedule_for_module() → returns schedule data
5. Chat.py passes this to LLM for natural language response
"""

import re
from config import WINTER_SEMS, SUMMER_SEMS, ELECTIVE_SEMS, is_semester_active, WINTER_SEMESTERS, SUMMER_SEMESTERS
from typing import Dict, Any, Optional

def extract_semester_criteria(query: str) -> Dict[str, Optional[Any]]:
    query = query.lower()
    filters: Dict[str, Optional[Any]] = {
        "semester_num": None,
        "season": None,
    }

    # Word mapping
    word_map = {
        "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7, "final": 7, "last": 7
    }

    for word, num in word_map.items():
        if word in query:
            filters["semester_num"] = num
            break
            
    if not filters["semester_num"]:
        match = re.search(r'semester\s+(\d+)', query)
        if match:
            filters["semester_num"] = int(match.group(1))

    if "winter" in query: filters["season"] = "winter"
    if "summer" in query: filters["season"] = "summer"
    
    return filters

def get_modules_from_map(module_map, filters):
    results = []
    
    # New Logic: A student can take any module from the current or previous semesters
    # as long as the season (winter/summer) matches.
    
    target_semesters = []
    
    # Determine the target season based on the requested semester number or season name.
    season = None
    if filters["semester_num"]:
        if filters["semester_num"] in WINTER_SEMS:
            season = "winter"
        elif filters["semester_num"] in SUMMER_SEMS:
            season = "summer"
    elif filters["season"]:
        season = filters["season"]
        
    # Collect all valid semesters up to the requested one.
    if season == "winter":
        max_sem = filters["semester_num"] or max(WINTER_SEMS)
        target_semesters = [s for s in WINTER_SEMS if s <= max_sem]
    elif season == "summer":
        max_sem = filters["semester_num"] or max(SUMMER_SEMS)
        target_semesters = [s for s in SUMMER_SEMS if s <= max_sem]
    else: # If no season is specified, just use the semester number if available.
        if filters["semester_num"]:
            target_semesters = [filters["semester_num"]]

    for code, name in module_map.items():
        try:
            parts = code.split('_')
            if len(parts) < 2: continue
            
            identifier = parts[1].split('.')[0] # '1', '3', 'W', 'K'
            
            # 1. Standard Semesters (CI_1, CI_2...)
            if identifier.isdigit():
                if int(identifier) in target_semesters:
                    results.append(f"{code}: {name}")
            
            # 2. Electives (CI_W) and Key Competences (CI_K)
            # These are typically in semesters 4 and 5.
            elif identifier in ['W', 'K']:
                # Suggest electives if the student is in or past the elective semesters.
                if any(s in ELECTIVE_SEMS for s in target_semesters):
                    results.append(f"{code}: {name} (Elective/Key Competence)")
                    
        except Exception:
            continue
            
    return sorted(results)

def find_code_by_name(query, module_map):
    """
    Fuzzy search: If user says 'Physics', find 'CI_1.07 Physics...'
    Returns the Code (e.g., 'CI_1.07') or None.
    
    Uses a scoring system to find the best match:
    - Exact substring match gets highest priority
    - All query words present gets medium priority
    - Partial matches get lower priority
    """
    query = query.lower()
    
    # Ignore common stop words to avoid false positives
    stop_words = ["module", "course", "subject", "class", "what", "is", "who", "teaches", 
                  "when", "where", "time", "timing", "schedule", "day", "my", "the", "a", "an"]
    query_words = [w for w in query.split() if w not in stop_words and len(w) > 2]
    
    if not query_words:
        return None

    best_match = None
    best_score = 0
    
    for code, name in module_map.items():
        name_lower = name.lower()
        score = 0
        
        query_phrase = " ".join(query_words)

        # CRITICAL CHANGE: Prioritize exact or near-exact matches much more heavily.
        # This prevents "Data Science" from incorrectly matching "Software Engineering".
        
        # 1. Exact match (highest score)
        if query_phrase == name_lower:
            score = 200
        # 2. Query is a perfect substring of the name
        elif query_phrase in name_lower:
            score = 100
        # 3. All query words are in the name (less reliable)
        elif all(word in name_lower for word in query_words):
            score = 50 + len(query_words) * 10  # Bonus for more words
        # 4. Partial match (lowest score)
        else:
            matching_words = sum(1 for word in query_words if word in name_lower)
            if matching_words >= len(query_words) / 2 and len(query_words) > 1:
                score = matching_words * 10
        
        if score > best_score:
            best_score = score
            best_match = code
    
    return best_match if best_score > 0 else None


def detect_query_intent(query):
    """
    Analyzes the user's query to determine what type of information they're asking for.
    
    Returns a dictionary with:
    - intent: 'schedule' | 'modules_list' | 'module_info' | 'general'
    - details: Additional context about the query
    """
    query_lower = query.lower()
    
    # Schedule-related keywords
    # Note: we explicitly include "block" so questions like
    # "Are there any block dates?" are routed to schedule logic
    schedule_indicators = [
        'schedule', 'class', 'classes', 'when', 'time', 'timing', 'today', 
        'tomorrow', 'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        'what day', 'which day', 'room', 'where', 'professor', 'instructor', 'teacher',
        'block', 'block dates'
    ]
    
    # Module list keywords
    list_indicators = [
        'what modules', 'what subjects', 'what courses', 'list of', 'modules do i have',
        'subjects do i have', 'courses do i have', 'what do i study', 'my modules',
        'my subjects', 'my courses', 'curriculum'
    ]
    
    # Specific module info keywords
    info_indicators = [
        'tell me about', 'what is', 'who teaches', 'credits', 'ects', 'prerequisites',
        'entry requirements', 'description', 'workload', 'content', 'objectives'
    ]
    
    # Check for schedule queries
    if any(indicator in query_lower for indicator in schedule_indicators):
        return {
            'intent': 'schedule',
            'needs_semester': 'semester' in query_lower or any(d in query_lower for d in ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th']),
            'needs_day': 'today' in query_lower or any(d in query_lower for d in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']),
            'specific_module': None  # Will be filled by find_code_by_name
        }
    
    # Check for module list queries
    elif any(indicator in query_lower for indicator in list_indicators):
        return {
            'intent': 'modules_list',
            'needs_semester': True
        }
    
    # Check for specific module info queries
    elif any(indicator in query_lower for indicator in info_indicators):
        return {
            'intent': 'module_info',
            'specific_module': None  # Will be filled by find_code_by_name
        }
    
    # Default to general query
    else:
        return {
            'intent': 'general'
        }

def get_schedule_for_module(module_name, schedule_data, module_code=None):
    """
    Retrieves the schedule for a specific module.
    
    Args:
        module_name: The name of the module to search for
        schedule_data: The list of all schedule entries
        module_code: Optional module code (e.g., 'CI_3.02') to help narrow search
    
    Returns:
        List of schedule entries matching the module
    """
    results = []
    
    # If we have a module code, we can also search by the numeric part
    # e.g., CI_3.02 might have schedule code 8331
    search_terms = [module_name.lower()]
    
    for entry in schedule_data:
        entry_name = entry["module_name"].lower()
        
        # Check if any search term appears in the entry name
        if any(term in entry_name for term in search_terms):
            results.append(entry)
    
    return results


def get_schedule_for_day(semester, day, schedule_data):
    """
    Retrieves the schedule for a given semester and day.
    
    Also checks if the semester is active in the current season.
    
    Args:
        semester: The semester number (1-7)
        day: The day of the week (e.g., 'Monday', 'Tuesday', etc.)
        schedule_data: The list of all schedule entries
    
    Returns:
        tuple: (results, is_active, current_season)
            - results: List of schedule entries for that semester and day, sorted by start time
            - is_active: Boolean indicating if semester has classes in current season
            - current_season: String ('Winter' or 'Summer')
    """
    results = []
    
    # check if semester is active in current season
    is_active, current_season = is_semester_active(semester)
    
    # if semester is not active, return empty results with the metadata
    if not is_active:
        return [], is_active, current_season
    
    # Normalize day name (handle case variations)
    day_normalized = day.strip().capitalize()
    
    for entry in schedule_data:
        if entry["semester"] == semester and entry["day"] == day_normalized:
            results.append(entry)
    
    # Sort by start time for better readability
    results.sort(key=lambda x: x["start_time"])
    
    return results, is_active, current_season


def get_all_schedule_for_semester(semester, schedule_data):
    """
    Retrieves all schedule entries for a given semester.
    
    Args:
        semester: The semester number (1-7)
        schedule_data: The list of all schedule entries
    
    Returns:
        Dictionary mapping day names to lists of schedule entries
    """
    schedule_by_day = {
        'Monday': [],
        'Tuesday': [],
        'Wednesday': [],
        'Thursday': [],
        'Friday': []
    }
    
    for entry in schedule_data:
        if entry["semester"] == semester and entry["day"] in schedule_by_day:
            schedule_by_day[entry["day"]].append(entry)
    
    # Sort each day's schedule by start time
    for day in schedule_by_day:
        schedule_by_day[day].sort(key=lambda x: x["start_time"])
    
    return schedule_by_day