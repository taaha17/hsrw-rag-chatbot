"""
Quick test script to verify the chatbot logic without interactive loop
"""

import json
import datetime
from logic_engine import (
    extract_semester_criteria,
    get_modules_from_map,
    find_code_by_name,
    get_schedule_for_module,
    get_schedule_for_day,
    detect_query_intent
)

# Load data
with open('data/module_map.json', 'r') as f:
    module_map = json.load(f)
with open('data/class_schedule.json', 'r') as f:
    schedule_data = json.load(f)

# Test queries
test_cases = [
    "When is signals and systems class?",
    "What classes do I have today? I'm in 3rd semester.",
    "Which semester is physics offered?",
    "What modules do I have in 2nd semester?",
    "i want to know the class timing for signals and systems"
]

print("=" * 80)
print("TESTING QUERY UNDERSTANDING AND DATA RETRIEVAL")
print("=" * 80)

for query in test_cases:
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    # Step 1: Detect intent
    intent_info = detect_query_intent(query)
    print(f"Intent: {intent_info['intent']}")
    
    # Step 2: Try to find module
    module_code = find_code_by_name(query, module_map)
    if module_code:
        print(f"Module Found: {module_code} - {module_map[module_code]}")
        
        # Get schedule
        schedule = get_schedule_for_module(module_map[module_code], schedule_data, module_code)
        if schedule:
            print(f"Schedule Data:")
            for entry in schedule:
                print(f"  - {entry['day']} {entry['start_time']}-{entry['end_time']}")
                print(f"    Type: {entry['type']}, Professor: {entry['professor']}, Room: {entry['room']}")
    
    # Step 3: Extract semester criteria
    filters = extract_semester_criteria(query)
    if filters['semester_num']:
        print(f"Semester Detected: {filters['semester_num']}")
        
        # If it's a schedule query, get today's schedule
        if intent_info['intent'] == 'schedule':
            day = datetime.datetime.now().strftime('%A')
            print(f"Getting schedule for {day}...")
            day_schedule = get_schedule_for_day(filters['semester_num'], day, schedule_data)
            print(f"Found {len(day_schedule)} classes")
            for entry in day_schedule:
                print(f"  - {entry['start_time']}-{entry['end_time']}: {entry['module_name']}")
        
        # If it's a modules list query
        elif intent_info['intent'] == 'modules_list':
            modules = get_modules_from_map(module_map, filters)
            print(f"Modules in semester {filters['semester_num']}: {len(modules)}")
            for mod in modules[:5]:  # Show first 5
                print(f"  - {mod}")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
