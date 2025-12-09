# HSRW RAG Chatbot

An intelligent academic advisor chatbot for Infotronic Systems Engineering (ISE) students at Rhine-Waal University of Applied Sciences. The chatbot combines structured data (class schedules, module catalogs) with RAG (Retrieval-Augmented Generation) to provide accurate, contextual answers to student queries.

## Features

### 🎯 Smart Query Understanding
- **Intent Detection**: Automatically classifies queries into schedule, module list, module info, or general categories
- **Natural Language Processing**: Understands variations like "when is my physics class?", "what classes do I have today?", "which semester is signals and systems offered?"
- **Fuzzy Module Matching**: Finds modules even with partial names or common variations

### 📅 Class Schedule Queries
- Get schedule for specific modules (day, time, professor, room)
- View all classes for a specific day and semester
- Supports multiple class types: Lectures (L), Exercises (E), Labs (P), Combined (L&E)
- Accurate professor names and room locations

### 📚 Module Information
- List modules by semester (1-7) or season (winter/summer)
- Detailed module information: credits, prerequisites, content, learning objectives
- Elective and key competence module recommendations

### 🔍 Intelligent Data Routing
- Prioritizes structured schedule data over RAG retrieval for schedule queries
- Uses RAG for detailed module descriptions and general questions
- Hybrid search combining BM25 and vector similarity

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Query                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ detect_query_  │
                    │    intent()    │  ◄── Logic Engine
                    └────────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────┐  ┌──────────────┐  ┌──────────┐
      │ Schedule │  │ Module List  │  │   RAG    │
      │  Data    │  │ (JSON)       │  │ Retrieval│
      └────┬─────┘  └──────┬───────┘  └─────┬────┘
           │               │                 │
           └───────────────┼─────────────────┘
                           ▼
                    ┌─────────────┐
                    │     LLM     │
                    │  (Ollama)   │
                    └──────┬──────┘
                           │
                           ▼
                   ┌──────────────┐
                   │   Response   │
                   └──────────────┘
```

## Setup

**Prerequisite:** Python 3.11.9+ and Ollama installed

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
    
2.  **Activate the virtual environment:**
    -   **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
        
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Setup Ollama:**
    -   Download from [ollama.ai](https://ollama.ai/)
    -   Pull the required models:
        ```bash
        ollama pull all-minilm
        ollama pull llama3.2
        ```
        
5.  **Add PDF documents:**
    -   Place your PDF files into the `data/` directory:
      - `ISE_CS.pdf` - Class Schedule
      - `ISE_MH.pdf` - Module Handbook
      - `ISE_ER_*.pdf` - Examination Regulations

## Usage

1.  **Ingest the documents:**
    ```bash
    python ingest.py
    ```
    This will:
    - Parse class schedules and extract structured data
    - Extract module information from the handbook
    - Create embeddings and populate the vector database
    - Generate `class_schedule.json` and `module_map.json`

2.  **Run the chatbot:**
    ```bash
    python chat.py
    ```

3.  **Example Queries:**
    - "When is my Signals and Systems class?"
    - "What classes do I have today? I'm in 3rd semester."
    - "Which modules do I have in 2nd semester?"
    - "Tell me about the Data Science module"
    - "Who teaches Machine Learning?"
    - "What are the prerequisites for Embedded Systems?"

## Project Structure

```
hsrw-rag-chatbot/
├── chat.py              # Main chatbot interface with smart query routing
├── ingest.py            # Document parsing and vector DB creation
├── logic_engine.py      # Query understanding and data filtering
├── config.py            # Configuration and constants
├── utils.py             # Helper utilities (Ollama embeddings)
├── test_queries.py      # Test script for query logic validation
├── data/
│   ├── module_map.json      # Module code to name mappings
│   ├── class_schedule.json  # Structured schedule data
│   └── *.pdf                # Source documents
├── chroma_db/           # Vector database (auto-generated)
└── requirements.txt     # Python dependencies
```

## Key Improvements (December 2025)

### 🔧 Enhanced PDF Parsing
- **Multi-line module names**: Correctly handles names spanning multiple lines (e.g., "Physics: Mechanics, Electricity and\nMagnetism")
- **Complete professor names**: Properly concatenates split names (e.g., "Prof. Dr. Große-\nKampmann")
- **State machine parser**: More robust than regex-only approach, handles edge cases

### 🧠 Smarter Query Understanding
- **Intent classification**: `detect_query_intent()` categorizes queries for appropriate routing
- **Improved module matching**: Scoring system with exact substring, all-words, and partial matching
- **Day/semester extraction**: Handles natural language like "today", "3rd semester", "winter"

### 💬 Better LLM Prompting
- **Structured data emphasis**: Clear visual separation and mandatory instructions
- **No hallucinations**: Explicit instructions to use provided data, not generic advice
- **Context-aware prompts**: Different system prompts for schedule vs. general queries

## Development Notes

### Adding New Features
- **New query types**: Add to `detect_query_intent()` in `logic_engine.py`
- **Additional data sources**: Add parsing logic in `ingest.py`
- **Custom prompts**: Modify `generate_chat_response()` in `chat.py`

### Configuration
All configurable values are in `config.py`:
- Ollama models and URL
- File paths and directory structure
- Semester/season mappings
- Class type definitions

### Testing
Run `test_queries.py` to validate query understanding without launching the full chatbot:
```bash
python test_queries.py
```
