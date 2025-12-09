"""
Configuration File - Central Settings for the Chatbot

All configurable values are defined here to make the system easy to maintain.
If you need to change file paths, model names, or semester mappings, do it here.

Key Configurations:
- File paths: Where to find data, vector DB, and generated JSON files
- Ollama settings: Which models to use and where the server is
- Semester mappings: Which semesters are winter/summer, when electives are offered
- Class type definitions: What L, E, P, etc. mean

To modify behavior:
- Change CHAT_MODEL to use a different LLM (e.g., 'llama3.3', 'mistral')
- Adjust WINTER_SEMS/SUMMER_SEMS if curriculum changes
- Update DAY_NAMES if supporting other languages
"""

import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
MODULE_MAP_PATH = os.path.join(DATA_FOLDER, "module_map.json")
CLASS_SCHEDULE_PATH = os.path.join(DATA_FOLDER, "class_schedule.json")

# --- OLLAMA ---
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "all-minilm"
CHAT_MODEL = "llama3.2"

# --- RAG ---
# Regex: Matches "CI_X.YY" followed by text
MODULE_PATTERN = r"^(CI_[W|K|\d]\.\d{2})\s+(.+)$"

# --- LOGIC ENGINE ---
# Semester to Season mapping for ISE
WINTER_SEMS = [1, 3, 5, 7]
SUMMER_SEMS = [2, 4, 6]
ELECTIVE_SEMS = [4, 5]

# Day name mappings (for normalization)
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Class type explanations
CLASS_TYPES = {
    'L': 'Lecture (Vorlesung)',
    'E': 'Exercise/Tutorial (Übung)',
    'P': 'Practical/Lab (Praktikum)',
    'L&E': 'Combined Lecture and Exercise',
    'PT': 'Lab Project',
    'SL': 'Self-Learning'
}
