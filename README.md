# zero - ai academic advisor

ai-powered chatbot for rhine-waal university students to get instant answers about courses, schedules, and academic requirements.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## features

- **smart query understanding** - detects what you're asking (schedule, modules, general info)
- **natural language** - ask in your own words, no special syntax needed
- **schedule information** - class times, rooms, professors, building locations
- **academic guidance** - module listings, prerequisites, credits, course descriptions
- **modern chat interface** - web-based, dark theme, conversation history
- **context-aware** - remembers your semester and degree program
- **local & private** - runs on your machine, no data sent externally

---

## quick start

### 1. prerequisites

- python 3.11 or higher
- ollama installed ([download](https://ollama.ai/))

### 2. installation

```bash
# clone repository
git clone https://github.com/taaha17/hsrw-rag-chatbot.git
cd hsrw-rag-chatbot

# create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # windows powershell
# source venv/bin/activate    # linux/mac

# install dependencies
pip install -r requirements.txt

# pull ollama models
ollama pull all-minilm
ollama pull llama3.2
```

### 3. run

```bash
python run.py
```

this will:
- check if data exists
- run ingestion if needed (processes pdfs, creates vector database)
- launch web interface at http://localhost:7860

**options:**
- `python run.py --force` - force re-ingest data
- `python run.py --ingest` - only ingest, don't start app

---

## usage

### example questions

**schedule queries:**
- "what classes do i have today?"
- "when is my signals and systems lecture?"
- "show me my tuesday schedule"

**module information:**
- "what modules are in semester 3?"
- "tell me about machine learning"
- "what are the prerequisites for embedded systems?"

**general questions:**
- "which semester is the internship?"
- "how many ects is physics?"

---

## project structure

```
hsrw-rag-chatbot/
├── app.py              # gradio web interface
├── chat.py             # llm interaction & response generation
├── logic_engine.py     # query intent detection & data filtering
├── ingest.py           # pdf processing & vector db creation
├── config.py           # centralized configuration
├── utils.py            # custom ollama embeddings
├── run.py              # master startup script
├── data/               # parsed json data (module map, schedules)
├── chroma_db/          # vector database
└── logs/               # application logs
```

---
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

### Components

#### 🖥️ **Frontend** (`app.py`)
- **Gradio interface**: Modern, responsive web UI
- **User context management**: Degree program and semester tracking
- **Rich formatting**: Emoji, tables, markdown
- **Example queries**: Guided user experience

#### 🧠 **Logic Engine** (`logic_engine.py`)
- **Intent detection**: Classifies query type (schedule, module_list, module_info, general)
- **Fuzzy matching**: Finds modules even with partial names
- **Schedule retrieval**: By module, day, or semester
- **Hybrid routing**: Structured data + RAG

#### 💬 **Chat System** (`chat.py`)
- **Conversation memory**: Maintains chat history
- **LLM integration**: Ollama llama3.2
- **System prompts**: Context-aware instructions
- **Response generation**: Natural, helpful answers

#### 🔍 **RAG Pipeline** (`ingest.py`)
- **PDF parsing**: pdfplumber for text extraction
- **Schedule parser**: Regex-based structured extraction
- **Module mapping**: JSON storage for fast lookups
- **Vector embeddings**: ChromaDB + all-minilm
- **Hybrid retrieval**: BM25 + semantic search

#### ⚙️ **Configuration** (`config.py`)
- **Degree programs**: Extensible structure
- **Class type decoder**: L→Lecture, E→Exercise, P→Practical
- **Room formatter**: "Building 1, Ground Floor, Room 215"
## how it works

### architecture

```
user question
    ↓
intent detection (what is the user asking?)
    ↓
    ├── schedule query → use json data (exact times, rooms)
    ├── module list → use json data (semester-specific)
    └── module info → use rag search (detailed descriptions)
    ↓
context building
    ↓
llm generation (ollama/llama3.2)
    ↓
response to user
```

### technologies

- **python 3.11** - core language
- **gradio 6.0** - web interface
- **langchain** - rag framework
- **chromadb** - vector database
- **ollama** - local llm server
  - llama3.2 (chat model)
  - all-minilm (embeddings)
- **pdfplumber** - pdf text extraction

### key concepts

**rag (retrieval augmented generation)**
combines document search with llm generation to provide accurate, source-grounded answers

**intent detection**
determines what the user is asking for and routes to the appropriate data source

**hybrid search**
uses both keyword matching (bm25) and semantic search (vectors) for better retrieval

---

## advanced

### regenerate data
```bash
python run.py --force
```

### ingest only (no app launch)
```bash
python run.py --ingest
```

### check logs
```bash
# logs are in logs/ directory
logs/zero_chat.log     # conversation history
logs/zero_debug.log    # errors and warnings
logs/zero_prompts.log  # llm prompts for tuning
```

---

## troubleshooting

**"no module named 'pdfplumber'"**
```bash
pip install -r requirements.txt
```

**"ollama connection error"**
```bash
ollama serve
ollama list  # verify models installed
```

**"empty schedule returned"**
```bash
python run.py --force  # re-ingest data
```

---

## documentation

- **GUIDE.md** - comprehensive technical guide and interview preparation
- **requirements.txt** - python dependencies

---

## license

mit license - see license file

---

**built for rhine-waal university students**
