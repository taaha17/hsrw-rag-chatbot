"""
config.py - central configuration for the chatbot application

this file contains all configurable settings:
- file paths for data and database
- ollama model names and server url
- semester/schedule mappings
- logging setup

to change models or paths, edit the values here rather than throughout the codebase
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
MODULE_MAP_PATH = os.path.join(DATA_FOLDER, "module_map.json")
CLASS_SCHEDULE_PATH = os.path.join(DATA_FOLDER, "class_schedule.json")
LOGS_FOLDER = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_FOLDER, exist_ok=True)

# ollama configuration
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "all-minilm"
CHAT_MODEL = "llama3.2"

# rag settings
# regex pattern to match module codes like "CI_1.02 Module Name"
MODULE_PATTERN = r"^(CI_[W|K|\d]\.\d{2})\s+(.+)$"

# semester organization
# winter semesters start in october, summer semesters in march
WINTER_SEMS = [1, 3, 5, 7]
SUMMER_SEMS = [2, 4, 6]
ELECTIVE_SEMS = [4, 5]  # students choose electives in these semesters

# Day name mappings (for normalization)
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Class type explanations - Full names for user display
CLASS_TYPES = {
    'L': 'Lecture',
    'E': 'Exercise',
    'P': 'Practical',
    'L&E': 'Lecture & Exercise',
    'PT': 'Practical Training',
    'SL': 'Self-Learning'
}

# Degree programs - Structured for future expansion
DEGREE_PROGRAMS = {
    'ISE': {
        'name': 'Infotronic Systems Engineering',
        'code': 'ISE',
        'max_semesters': 7,
        'schedule_file': 'ISE_CS.pdf',
        'handbook_file': 'ISE_MH.pdf'
    }
    # Future programs will be added here:
    # 'ME': {'name': 'Mechanical Engineering', ...},
    # 'EE': {'name': 'Electrical Engineering', ...},
}

# semester-season mapping
# winter semesters: 1, 3, 5, 7 (odd semesters)
# summer semesters: 2, 4, 6 (even semesters)
WINTER_SEMESTERS = [1, 3, 5, 7]
SUMMER_SEMESTERS = [2, 4, 6]


def get_current_season():
    """determine if current date is in winter or summer semester"""
    import pytz
    from datetime import datetime
    
    tz = pytz.timezone('Europe/Berlin')
    now = datetime.now(tz)
    month = now.month
    
    # winter semester: october (10) to march (3)
    # summer semester: april (4) to september (9)
    if month >= 10 or month <= 3:
        return 'Winter'
    else:
        return 'Summer'


def is_semester_active(semester: int) -> tuple[bool, str]:
    """
    check if given semester has classes in current season
    
    returns:
        (is_active, season) - tuple of boolean and current season name
    """
    current_season = get_current_season()
    
    if current_season == 'Winter':
        is_active = semester in WINTER_SEMESTERS
    else:  # Summer
        is_active = semester in SUMMER_SEMESTERS
    
    return is_active, current_season


def decode_class_type(type_code):
    """
    Convert class type shortcode to full name.
    Example: 'L' -> 'Lecture', 'E' -> 'Exercise'
    """
    return CLASS_TYPES.get(type_code, type_code)


def format_room_info(room_name, building=None, floor=None, room_num=None):
    """
    Format room information for user display.
    
    Args:
        room_name: Generic name like 'Hörsaal', 'Seminarraum'
        building: Building number (e.g., '01', '04')
        floor: Floor number (e.g., '00' = ground floor, '02' = 2nd floor)
        room_num: Room number (e.g., '215', '130')
    
    Returns:
        Formatted string like 'Building 1, Ground Floor, Room 215' or just 'Hörsaal' if no codes
    """
    # If we have building/floor/room codes, use them
    if building and floor and room_num:
        # Convert to integers for cleaner display
        building_int = int(building)
        floor_int = int(floor)
        
        # Special handling for ground floor
        if floor_int == 0:
            floor_str = "Ground Floor"
        else:
            floor_str = f"Floor {floor_int}"
        
        return f"{room_name}, Building {building_int}, {floor_str}, Room {room_num}"
    
    # Otherwise just return the room name
    return room_name


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging():
    """
    Configure comprehensive logging for Zero chatbot.
    
    Creates 3 separate log files:
    1. zero_chat.log - User conversations (queries, responses, context)
    2. zero_debug.log - Debug info, errors, warnings
    3. zero_prompts.log - LLM prompts and responses for refinement
    
    Each log file:
    - Rotates at 10MB (keeps 5 backups)
    - UTF-8 encoding for international characters
    - Timestamped entries
    
    To adjust log verbosity:
    - Change level=logging.INFO to logging.DEBUG for more detail
    - Change level=logging.WARNING for production (less noise)
    """
    
    # Formatter with timestamp, level, and message
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # === CHAT LOGGER (User conversations) ===
    chat_logger = logging.getLogger('zero.chat')
    chat_logger.setLevel(logging.INFO)
    chat_logger.propagate = False  # Don't send to root logger
    
    chat_handler = RotatingFileHandler(
        os.path.join(LOGS_FOLDER, 'zero_chat.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    chat_handler.setFormatter(detailed_formatter)
    chat_logger.addHandler(chat_handler)
    
    # === DEBUG LOGGER (Errors, warnings, system events) ===
    debug_logger = logging.getLogger('zero.debug')
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.propagate = False
    
    debug_handler = RotatingFileHandler(
        os.path.join(LOGS_FOLDER, 'zero_debug.log'),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    debug_handler.setFormatter(detailed_formatter)
    debug_logger.addHandler(debug_handler)
    
    # === PROMPT LOGGER (LLM prompts/responses for refinement) ===
    prompt_logger = logging.getLogger('zero.prompts')
    prompt_logger.setLevel(logging.INFO)
    prompt_logger.propagate = False
    
    prompt_handler = RotatingFileHandler(
        os.path.join(LOGS_FOLDER, 'zero_prompts.log'),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    prompt_handler.setFormatter(detailed_formatter)
    prompt_logger.addHandler(prompt_handler)
    
    # Also add console handler for errors (so they appear in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter('❌ %(levelname)s: %(message)s'))
    debug_logger.addHandler(console_handler)
    
    return chat_logger, debug_logger, prompt_logger


# Initialize loggers (called once at import)
chat_log, debug_log, prompt_log = setup_logging()

# Log initialization
debug_log.info("="*80)
debug_log.info("Zero chatbot logging initialized")
debug_log.info(f"Logs directory: {LOGS_FOLDER}")
debug_log.info("="*80)
