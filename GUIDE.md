# Zero Chatbot - Complete Technical Guide for Interview

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Design](#architecture--design)
3. [Technology Stack](#technology-stack)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Key Concepts](#key-concepts)
7. [Interview Preparation](#interview-preparation)

---

## Project Overview

### What is Zero?
zero is an ai-powered academic advisor chatbot designed specifically for university students. it helps students get instant answers about their courses, schedules, module information, and academic requirements.

### The Problem It Solves
students often struggle to find information about:
- when and where their classes are
- what modules they need to take each semester
- prerequisites for courses
- professor information and room locations
- understanding complex module handbooks

traditionally, this requires:
- digging through pdf documents
- emailing advisors and waiting for responses
- manually cross-referencing schedules and handbooks

**zero solves this by providing instant, conversational access to all this information.**

### Why This Approach?
instead of building a traditional database-driven system, this project uses:
1. **rag (retrieval augmented generation)** - combines document search with ai generation
2. **local llm** - runs entirely on your machine, no data sent to external servers
3. **smart query routing** - understands what you're asking and fetches the right data

this makes it accurate, fast, private, and easy to maintain.

---

## Architecture & Design

### High-Level Architecture

```
User Input (Web UI)
        ↓
  Intent Detection (what is the user asking?)
        ↓
   ┌────┴────┐
   ↓         ↓
Structured   RAG
  Data     Search
   ↓         ↓
   └────┬────┘
        ↓
  Context Building
        ↓
   LLM Generation
        ↓
  Response to User
```

### System Components

```
hsrw-rag-chatbot/
├── app.py              # gradio web interface
├── chat.py             # llm interaction & response generation  
├── logic_engine.py     # intent detection & data filtering
├── ingest.py           # pdf processing & vector db creation
├── config.py           # centralized settings
├── utils.py            # ollama embedding implementation
├── run.py              # startup script
├── data/               # parsed json data
├── chroma_db/          # vector database
└── logs/               # application logs
```

### Design Principles

**1. separation of concerns**
- `app.py` handles only ui logic
- `chat.py` manages llm communication
- `logic_engine.py` does query understanding
- `ingest.py` processes documents once

**2. structured data first, rag second**
- for factual queries (schedules, lists), use json data directly
- for detailed explanations, use rag to search pdfs
- this prevents llm hallucinations

**3. clear data flow**
- user query → intent detection → data retrieval → context building → llm → response
- each step is separate and testable

---

## Technology Stack

### Core Technologies

**python 3.11**
- modern, readable, excellent libraries for ai/ml
- strong typing support with type hints
- great for rapid development

**ollama**
- runs large language models locally
- supports multiple models (llama, mistral, etc.)
- provides both chat and embedding apis
- completely private - no data leaves your machine

**langchain**
- framework for building llm applications
- provides tools for document processing, embeddings, retrievers
- makes rag implementation simpler

**gradio**
- creates web uis with python code
- interactive chat interface
- no need for html/css/javascript

**chromadb**
- vector database for storing document embeddings
- enables semantic search over documents
- fast similarity search

### Why These Choices?

**why ollama instead of openai/claude?**
- privacy: university data stays local
- cost: no api fees
- control: can use any open-source model
- latency: no network calls to external apis

**why rag instead of just an llm?**
- accuracy: llm can cite actual documents
- up-to-date: as pdfs update, knowledge updates
- verifiable: can show sources
- reduces hallucinations: grounded in real data

**why gradio?**
- rapid prototyping: ui in minutes
- python-native: no separate frontend needed
- interactive: supports chat, file upload, buttons
- easy deployment: one command to run

---

## Core Components

### 1. ingest.py - document processing

**what it does:**
converts pdf documents into searchable data

**the process:**
```python
# step 1: load pdf
pdf → extract text with pdfplumber

# step 2: parse structured data
identify module codes (CI_1.02)
extract schedules (day, time, room)
create json files for quick lookup

# step 3: create vector embeddings
split documents into chunks
generate embeddings with all-minilm
store in chromadb for semantic search
```

**key functions:**
- `parse_module_handbook()` - extracts module metadata
- `parse_class_schedule()` - parses schedule tables
- `ingest_documents()` - orchestrates the entire pipeline

**challenges solved:**
- multi-line module names in pdfs
- distinguishing table headers from data
- handling split text across pages

### 2. logic_engine.py - query understanding

**what it does:**
figures out what the user is asking for and what data is relevant

**core functions:**

**`detect_query_intent(query)`**
determines if the user wants:
- schedule information ("when is my class?")
- module list ("what courses do i take?")
- module details ("tell me about physics")
- general information ("how many semesters?")

```python
# uses keyword matching
if any(word in query for word in ['when', 'time', 'schedule']):
    return 'schedule'
```

**`extract_semester_criteria(query)`**
extracts which semester the user is asking about:
```python
"second semester" → semester_num: 2
"winter semester" → season: winter
"5th semester modules" → semester_num: 5
```

**`find_code_by_name(query, module_map)`**
fuzzy matches module names:
```python
"signals and systems" → CI_3.02
"physics" → CI_1.07
```

**`get_schedule_for_module(module_name, schedule_data, code)`**
retrieves schedule entries for a specific module

**why this matters:**
without intent detection, we'd have to send everything to the llm, which:
- is slow (more tokens to process)
- costs more (if using paid apis)
- is less accurate (more context = more confusion)

by understanding intent first, we retrieve only relevant data.

### 3. chat.py - llm interaction

**what it does:**
communicates with the ollama llm to generate natural language responses

**the response generation flow:**
```python
def generate_chat_response(context, history, question, hardcoded_list=None):
    # 1. build system prompt with instructions
    system_prompt = """
    you are an academic advisor...
    here is the official data: {context}
    answer based on this data only
    """
    
    # 2. construct messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": previous_user_message},
        {"role": "assistant", "content": previous_bot_response},
        {"role": "user", "content": current_question}
    ]
    
    # 3. send to ollama
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3.2", "messages": messages}
    )
    
    # 4. extract and return response
    return response.json()["message"]["content"]
```

**key features:**
- **system prompts** - instructs the llm on how to behave
- **conversation history** - maintains context across messages
- **context injection** - inserts retrieved data into prompts
- **error handling** - gracefully handles api failures

**prompt engineering techniques used:**
```python
# technique 1: explicit constraints
"ONLY use information from the provided documents"

# technique 2: structured instructions
"1. read the schedule data
 2. extract relevant information
 3. present it clearly"

# technique 3: examples
"good response: 'your physics class is monday 10-12'
 bad response: 'i don't have access to schedules'"
```

### 4. app.py - web interface

**what it does:**
provides the gradio-based chat interface

**key features:**
- degree program dropdown
- semester selection
- chat history display
- message input
- clickable follow-up suggestions

**the interaction flow:**
```python
def chat_with_zero(message, history, degree, semester):
    # 1. detect what user is asking
    intent = detect_query_intent(message)
    
    # 2. route to appropriate data source
    if intent == 'schedule':
        context = get_schedule_data(...)
    elif intent == 'module_info':
        context = rag_search(...)
    
    # 3. generate response
    response = generate_chat_response(context, history, message)
    
    # 4. update history and return
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history
```

**gradio 6.x specific handling:**
gradio's chatbot component uses a multimodal format where content can be:
```python
# string format (simple)
"hello"

# multimodal format (gradio 6.x)
[{"text": "hello", "type": "text"}]
```

we handle this with `extract_text_from_message()` to ensure compatibility.

### 5. utils.py - ollama embeddings

**what it does:**
implements a custom embeddings class for ollama

**why custom implementation?**
langchain's official ollama client had:
- timeout issues with large documents
- unclear error messages
- less control over batching

our custom implementation:
```python
class HttpOllamaEmbeddings(Embeddings):
    def embed_query(self, text):
        # single text → embedding vector
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "all-minilm", "prompt": text}
        )
        return response.json()["embedding"]
    
    def embed_documents(self, texts):
        # multiple texts → list of embeddings
        return [self.embed_query(text) for text in texts]
```

**how embeddings work:**
```
text: "physics lecture" 
  ↓
embedding model (all-minilm)
  ↓
vector: [0.23, -0.45, 0.67, ...] (384 dimensions)
```

these vectors capture semantic meaning, allowing us to find similar documents.

---

## Data Flow

### startup flow

```
1. run.py executes
   ↓
2. checks if data exists (module_map.json, class_schedule.json)
   ↓
3. if missing: runs ingest.py
   ↓
4. loads gradio interface (app.py)
   ↓
5. initializes:
   - loads json data into memory
   - connects to chromadb
   - creates bm25 and vector retrievers
   - combines into ensemble retriever
   ↓
6. app ready at http://localhost:7860
```

### query processing flow

```
user: "when is my signals class?"
  ↓
app.py: receives message
  ↓
logic_engine.detect_query_intent()
  → intent: 'schedule'
  ↓
logic_engine.find_code_by_name("signals", module_map)
  → code: 'CI_3.02'
  ↓
logic_engine.get_schedule_for_module(CI_3.02, schedule_data)
  → schedule: {day: Monday, time: 10:00-11:30, ...}
  ↓
chat.generate_chat_response(schedule_data, history, query)
  ↓
  → builds context with schedule
  → sends to ollama llm
  → receives: "your signals class is monday 10:00-11:30..."
  ↓
app.py: displays response to user
```

### rag search flow

```
user: "tell me about machine learning module"
  ↓
intent: 'module_info'
  ↓
ensemble_retriever.invoke("machine learning")
  ↓
  → bm25 search (keyword matching)
  → vector search (semantic similarity)
  → combines and ranks results
  ↓
retrieves top 5 relevant document chunks
  ↓
chat.format_module_details_from_rag(chunks)
  → structures information (credits, prerequisites, etc.)
  ↓
sends to llm with structured template
  ↓
response with module details
```

---

## Key Concepts

### 1. RAG (Retrieval Augmented Generation)

**the problem rag solves:**
llms have training data cutoffs and can hallucinate. they don't know specific details about your university.

**how rag works:**
```
traditional llm:
question → llm → answer (may hallucinate)

rag approach:
question → search documents → relevant chunks → llm + chunks → grounded answer
```

**our rag implementation:**
```python
# step 1: create embeddings of all documents
documents → chunks → embeddings → store in chromadb

# step 2: at query time
user query → embed query → search chromadb → top k similar chunks

# step 3: augment llm prompt
prompt = f"""
given these documents:
{retrieved_chunks}

answer this question: {user_query}
"""
```

**why ensemble retriever?**
we combine two search methods:
- **bm25 (keyword search)** - good for exact matches ("CI_3.02", "physics")
- **vector search (semantic)** - good for concepts ("course about ai", "math class")

```python
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]  # equal weighting
)
```

### 2. Embeddings & Vector Databases

**what are embeddings?**
embeddings convert text into numerical vectors that capture meaning.

**example:**
```
"physics lecture"    → [0.2, 0.5, -0.3, ...]
"physics class"      → [0.21, 0.48, -0.29, ...] (similar!)
"cooking recipe"     → [-0.8, 0.1, 0.9, ...]  (different!)
```

**how similarity search works:**
```python
# 1. user asks: "quantum mechanics"
query_vector = embed("quantum mechanics")

# 2. calculate similarity to all stored vectors
similarities = []
for doc_vector in database:
    similarity = cosine_similarity(query_vector, doc_vector)
    similarities.append(similarity)

# 3. return top k most similar
top_results = sorted(similarities, reverse=True)[:5]
```

**why chromadb?**
- stores embeddings efficiently
- handles similarity search fast
- persists to disk
- simple api

### 3. Prompt Engineering

**what is prompt engineering?**
crafting instructions for llms to get desired outputs.

**techniques we use:**

**1. system prompts**
set the llm's role and behavior:
```python
system_prompt = """
you are an academic advisor for rhine-waal university.

rules:
- only answer using provided university documents
- if information isn't in the documents, say so
- be concise but complete
- use friendly, helpful tone
"""
```

**2. context injection**
provide relevant data directly:
```python
f"""
here is the official schedule:
- monday 10:00-12:00: physics (prof. smith, room 215)

student question: when is physics?
"""
```

**3. structured output requests**
tell the llm how to format responses:
```python
"""
present module information in this structure:
1. basic information (code, credits, semester)
2. course structure (lectures, exercises, labs)
3. learning outcomes
4. prerequisites
"""
```

**4. few-shot examples**
show the llm what good responses look like:
```python
"""
example good response:
'your signals class is on monday from 10:00-11:30 in hörsaal 1 with prof. dr. zimmer.'

example bad response:
'i don't have access to schedule information.' ❌
"""
```

### 4. Intent Detection

**why detect intent?**
different questions need different data sources:

```python
"when is my class?" → need schedule data
"what modules do i have?" → need module list
"tell me about physics" → need rag search
```

**how we detect intent:**
```python
def detect_query_intent(query):
    query = query.lower()
    
    # schedule intent
    if any(word in query for word in ['when', 'time', 'schedule', 'day']):
        return 'schedule'
    
    # module list intent  
    if any(phrase in query for phrase in ['what modules', 'which courses', 'my subjects']):
        return 'modules_list'
    
    # module info intent
    if any(word in query for word in ['about', 'tell me', 'describe', 'explain']):
        return 'module_info'
    
    # default
    return 'general'
```

**routing based on intent:**
```python
if intent == 'schedule':
    # use structured json data
    context = json.dumps(schedule_data)
elif intent == 'module_info':
    # use rag search
    docs = ensemble_retriever.invoke(query)
    context = format_docs(docs)
```

### 5. Conversation History

**why maintain history?**
enables natural follow-up questions:
```
user: "when is physics?"
bot: "monday 10-12"
user: "who teaches it?"  ← needs to remember "it" = physics
bot: "prof. smith"
```

**how history is stored:**
```python
history = [
    {"role": "user", "content": "when is physics?"},
    {"role": "assistant", "content": "monday 10-12 with prof. smith"},
    {"role": "user", "content": "who teaches it?"},
    {"role": "assistant", "content": "prof. smith teaches physics"}
]
```

**sending history to llm:**
```python
messages = [
    {"role": "system", "content": system_prompt},
    *history,  # all previous messages
    {"role": "user", "content": new_question}
]
```

**important:** only include user and assistant messages in history, not repeated system prompts.

---

## Interview Preparation

### Technical Questions You Might Face

**1. "why did you choose rag over fine-tuning the llm?"**

**answer:**
fine-tuning requires:
- large amounts of training data
- expensive gpu resources
- retraining when data changes
- risk of catastrophic forgetting

rag advantages:
- works with small datasets
- updates in real-time (update pdfs, no retraining)
- more transparent (can see what documents were used)
- cheaper and faster to implement

**2. "how do you prevent llm hallucinations?"**

**answer:**
several strategies:
1. **strict system prompts** - explicitly instruct to only use provided data
2. **structured data first** - for factual queries (schedules), use json directly
3. **rag grounding** - provide actual documents in context
4. **prompt engineering** - show examples of good vs bad responses
5. **validation** - can compare llm output against source documents

**3. "what if the vector database becomes too large?"**

**answer:**
several optimization strategies:
1. **chunking strategy** - optimize chunk size (currently 1000 chars)
2. **metadata filtering** - filter by semester, module type before searching
3. **hybrid search** - bm25 for initial filtering, vectors for final ranking
4. **hierarchical storage** - summary vectors + detailed vectors
5. **regular cleanup** - remove outdated document versions

**4. "how do you handle ambiguous queries?"**

**answer:**
1. **intent detection** determines query type
2. **context extraction** pulls relevant user info (semester, degree)
3. **smart defaults** - if semester not mentioned, use user's selected semester
4. **clarifying questions** - llm can ask "which semester?" if needed
5. **fuzzy matching** - handle typos in module names

**5. "why use a local llm instead of gpt-4?"**

**answer:**
**privacy:** university data sensitive, shouldn't leave campus
**cost:** no api fees, unlimited queries
**latency:** no network calls, faster responses
**control:** can choose any model, customize behavior
**offline capability:** works without internet

trade-offs:
- smaller models less capable than gpt-4
- need local hardware (gpu)
- model management overhead

for university use case, benefits outweigh drawbacks.

### Demonstrating Your Knowledge

**talking about your code decisions:**

**good approach:**
"i used an ensemble retriever combining bm25 and vector search because bm25 handles exact matches like module codes well, while vector search captures semantic meaning for descriptive queries."

**avoid:**
"i just copied this from the langchain documentation."

**showing problem-solving:**

**good approach:**
"i noticed gradio 6.x changed its message format to support multimodal content, so i created `extract_text_from_message()` to handle both the old string format and new list format."

**avoid:**
"i don't know why that function is there."

**discussing trade-offs:**

**good approach:**
"i chose to store schedule data as json rather than in the vector db because schedule queries need exact, structured data - embeddings are better for semantic search of unstructured text."

**avoid:**
"i just made it work."

### Handling Questions About AI Tools

**if asked: "did you use ai coding assistants?"**

**honest approach:**
"yes, i used ai tools for boilerplate code and debugging. but i understand every line - i can explain the rag pipeline, the intent detection logic, and the prompt engineering techniques. the architecture and design decisions were mine. ai helped me implement faster, but i guided the implementation."

**what matters:**
- you understand the code
- you can explain the concepts
- you can modify and extend it
- you know why decisions were made

**follow-up they might ask:**
"ok, explain how the ensemble retriever works"

**your answer:**
"the ensemble retriever combines two search methods - bm25 for keyword matching and vector similarity for semantic search. when a query comes in, both retrievers independently return their top results. the ensemble then merges them using weighted scores - currently 50/50. this catches both exact term matches like 'ci_3.02' and conceptual matches like 'machine learning class'."

### Common Interview Topics

**1. system design**
be ready to diagram the architecture, explain component interactions, and discuss scalability.

**2. data structures**
know how vectors are stored, how similarity search works, what indexes chromadb uses.

**3. machine learning basics**
understand embeddings, similarity metrics (cosine, euclidean), what transformers are.

**4. api design**
explain the ollama api, http requests, json formatting, error handling.

**5. production readiness**
discuss logging, error handling, testing, deployment, monitoring.

### Questions to Ask Them

**showing interest and understanding:**

1. "what scale do you expect the chatbot to handle? how many students would use it simultaneously?"

2. "are there plans to expand beyond ise to other programs? how would multi-program support work?"

3. "what's your data update frequency? how often do course schedules and handbooks change?"

4. "do you have preferences for deployment - on-premise servers, cloud, or hybrid?"

5. "are there compliance requirements i should know about? gdpr, student data protection?"

### Discussing Improvements

**be ready to suggest enhancements:**

**"if i had more time, i would add:"**
1. **multi-language support** - german/english toggle
2. **personalization** - remember user preferences
3. **calendar integration** - export schedules to google calendar
4. **email notifications** - remind about deadlines
5. **analytics dashboard** - track popular queries
6. **feedback mechanism** - students rate answer quality
7. **admin interface** - update data without re-ingesting
8. **mobile app** - native ios/android versions

**"to improve accuracy, i would:"**
1. **fine-tune embeddings** - train on university-specific vocabulary
2. **expand rag corpus** - include syllabi, past exam questions
3. **add fact verification** - cross-check llm outputs against source
4. **implement caching** - common queries cached for speed
5. **a/b testing** - test different prompts, models

### Red Flags to Avoid

**don't say:**
❌ "i don't know how that part works"
❌ "ai wrote all of this"
❌ "i just copied from stack overflow"
❌ "i haven't tested it much"
❌ "it works on my machine"

**instead say:**
✅ "let me walk through that component"
✅ "i used ai for boilerplate, but i understand the implementation"
✅ "i researched best practices and adapted them"
✅ "i've tested with [specific scenarios]"
✅ "i've documented deployment steps for different environments"

### Final Interview Tips

**1. be honest**
if you don't know something, say so - then explain how you'd find out.

**2. show enthusiasm**
talk about what you learned building this, what was challenging, what you're proud of.

**3. think aloud**
when answering technical questions, verbalize your thought process.

**4. connect to their needs**
relate your project to their university's specific requirements.

**5. be ready to code**
they might ask you to add a feature live - know your codebase well.

**6. highlight learning**
show you can pick up new technologies quickly (you learned ollama, langchain, gradio, rag).

**7. discuss collaboration**
mention how you'd work with their team, document your work, handle feedback.

### Practice Questions

**review these before the interview:**

1. walk me through the entire system architecture
2. explain how a user query becomes a response
3. what is rag and why use it?
4. how do embeddings work?
5. what challenges did you face building this?
6. how would you scale this to 10,000 users?
7. how do you ensure data privacy?
8. explain your testing strategy
9. what would you improve given more time?
10. how do you stay current with ai developments?

---

## Quick Reference

### key files cheat sheet

```
app.py           → web ui, user interaction
chat.py          → llm communication, response generation
logic_engine.py  → intent detection, data filtering
ingest.py        → pdf processing, vector db creation
config.py        → settings, paths, constants
utils.py         → custom ollama embeddings
run.py           → startup script
```

### important concepts

**rag** - retrieval augmented generation, combine search + llm
**embeddings** - text → vectors that capture meaning
**vector db** - stores embeddings, enables similarity search
**intent** - what user is asking for (schedule, modules, info)
**context** - relevant data injected into llm prompts
**system prompt** - instructions that set llm behavior

### ollama models

**all-minilm** - lightweight embedding model, 384 dimensions
**llama3.2** - 7b parameter chat model, instruction-tuned

### tech stack summary

- python 3.11
- ollama (local llm server)
- langchain (rag framework)
- chromadb (vector database)
- gradio (web ui)
- pdfplumber (pdf extraction)

---

## Final Thoughts

**you've built something impressive:**
- a production-ready rag application
- proper architecture with separation of concerns
- handles real-world data (messy pdfs, multi-line entries)
- user-friendly interface
- runs entirely locally for privacy

**what makes your project stand out:**
- hybrid search (bm25 + vectors)
- intent-based routing (structured data vs rag)
- proper prompt engineering
- conversation history handling
- clean, maintainable code

**you're ready for this interview.**

**remember:**
- understand concepts, not just code
- be honest about what you know
- show enthusiasm for learning
- connect your work to their needs
- think aloud when problem-solving

**good luck! 🚀**
