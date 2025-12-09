import re

def extract_semester_criteria(query):
    """
    Returns a dictionary of filters based on the user's query.
    """
    query = query.lower()
    filters = {
        "semester_num": None,
        "season": None,  # 'winter' or 'summer'
        "is_last": False,
        "is_elective": False
    }

    # 1. Word to Number Mapping
    word_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7,
        "final": 7, "last": 7
    }

    # 2. Find Semester Number
    for word, num in word_map.items():
        if word in query:
            filters["semester_num"] = num
            break
            
    # Fallback: look for "semester 3" digits
    if not filters["semester_num"]:
        match = re.search(r'semester\s+(\d+)', query)
        if match:
            filters["semester_num"] = int(match.group(1))

    # 3. Detect Season
    if "winter" in query:
        filters["season"] = "winter"
    elif "summer" in query:
        filters["season"] = "summer"

    return filters

def get_modules_from_map(module_map, filters):
    """
    Filters the loaded JSON map based on the criteria.
    """
    results = []
    
    # Define which semesters belong to which season
    season_map = {
        "winter": [1, 3, 5, 7],
        "summer": [2, 4, 6]
    }

    target_semesters = []
    
    # A. If explicit semester is asked (e.g. "3rd")
    if filters["semester_num"]:
        target_semesters = [filters["semester_num"]]
    
    # B. If season is asked (e.g. "Winter modules")
    elif filters["season"]:
        target_semesters = season_map[filters["season"]]

    # Search the map
    for code, name in module_map.items():
        # Code format: CI_3.02 -> split by _ and .
        # parts[0] = CI
        # parts[1] = 3.02
        try:
            # Extract the middle part (Semester indicator)
            # CI_3.02 -> '3'
            # CI_W.01 -> 'W'
            semester_char = code.split('_')[1].split('.')[0]
            
            if semester_char.isdigit():
                sem_int = int(semester_char)
                if sem_int in target_semesters:
                    results.append(f"{code}: {name}")
            
            # Handle special cases if needed (Electives usually don't have fixed semesters in code)
            
        except:
            continue
            
    return sorted(results)