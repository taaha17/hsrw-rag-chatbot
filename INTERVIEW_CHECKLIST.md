# interview checklist

## before the interview

### ✅ test everything works
- [ ] run `python run.py` and verify app launches
- [ ] test multi-turn conversation (ask 2-3 follow-up questions)
- [ ] test "what classes do i have today?" with current date
- [ ] test schedule query: "when is signals class?"
- [ ] test module query: "tell me about physics"
- [ ] verify all logs are writing to logs/ directory

### ✅ review key files
- [ ] read GUIDE.md completely (your technical reference)
- [ ] understand app.py (web interface, gradio event handlers)
- [ ] understand chat.py (llm communication, prompt engineering)
- [ ] understand logic_engine.py (intent detection, data routing)
- [ ] understand ingest.py (pdf parsing, vector db creation)
- [ ] understand config.py (settings, paths, logging setup)

### ✅ understand core concepts
- [ ] what is rag and why use it?
- [ ] how do embeddings work?
- [ ] what is chromadb and why use it?
- [ ] why ensemble retriever (bm25 + vectors)?
- [ ] what is intent detection and why needed?
- [ ] how does conversation history work?
- [ ] why use local llm (ollama) vs gpt-4?

### ✅ know the data flow
- [ ] user query → intent detection → data retrieval → context building → llm → response
- [ ] understand when json data is used vs rag search
- [ ] know how gradio 6.x multimodal format is handled
- [ ] understand how system prompts are constructed

### ✅ prepare to discuss decisions
- [ ] why separate schedule json from rag?
- [ ] why hybrid search (bm25 + vectors)?
- [ ] why structured data first approach?
- [ ] how did you handle gradio 6.x message format change?
- [ ] why three separate log files?

---

## interview preparation

### practice questions

**technical architecture:**
1. "walk me through the complete system architecture"
   - start with user query, end with response
   - mention each component: app.py, logic_engine, chat.py, ollama
   
2. "explain how rag works in your project"
   - document chunking → embeddings → vector storage
   - query embedding → similarity search → top k results
   - context injection into llm prompt

3. "what is the difference between bm25 and vector search?"
   - bm25: keyword matching, good for exact terms
   - vectors: semantic similarity, good for concepts
   - ensemble: combines both for better results

**problem-solving:**
1. "what was the most challenging bug you fixed?"
   - gradio 6.x multimodal format issue
   - explain the error, root cause, solution
   
2. "how do you prevent llm hallucinations?"
   - use structured data for factual queries
   - strict system prompts
   - rag grounding with actual documents
   - prompt engineering techniques

3. "how would you scale this to 10,000 concurrent users?"
   - horizontal scaling with load balancer
   - redis caching for common queries
   - query queue with background workers
   - cdn for static assets
   - separate read replicas for vector db

**code walkthrough:**
be ready to open any file and explain:
- what it does
- why it's structured this way
- how it interacts with other components

---

## key talking points

### what makes your project strong

1. **proper architecture**
   - clean separation of concerns
   - intent-based routing
   - structured data prioritized over llm guessing

2. **production-ready features**
   - comprehensive logging (chat, debug, prompts)
   - error handling throughout
   - conversation history management
   - multimodal format handling

3. **smart design decisions**
   - hybrid search for better retrieval
   - local llm for privacy and control
   - auto-detection in run.py
   - timezone awareness for "today" queries

4. **real problem solved**
   - students struggle with finding course info
   - manual pdf searching is slow
   - your chatbot makes it instant and conversational

### what you learned building this

- rag architecture and implementation
- working with local llms (ollama)
- vector databases and embeddings
- prompt engineering techniques
- gradio web interface framework
- python async and event handling
- parsing complex pdf documents
- logging and debugging strategies

---

## demo preparation

### have these ready to show

1. **quick demo**
   - launch app: `python run.py`
   - ask schedule question
   - ask module question
   - show follow-up works

2. **code tour**
   - open config.py, show clean organization
   - open logic_engine.py, explain intent detection
   - open chat.py, show prompt engineering
   - open app.py, show extract_text_from_message fix

3. **logs**
   - open logs/zero_chat.log, show conversation tracking
   - open logs/zero_debug.log, show error handling
   - explain why three separate logs

---

## questions to ask them

show you're thinking about their needs:

1. "what scale do you expect - how many students would use this?"
2. "are there plans to expand to other degree programs?"
3. "how often do course schedules and handbooks update?"
4. "what deployment environment do you have - on-premise or cloud?"
5. "are there specific compliance requirements for student data?"
6. "what features do students request most often?"

---

## potential improvements to discuss

show you're thinking ahead:

**short-term (easy wins):**
- add more degree programs (just add pdfs, re-ingest)
- calendar export (ics file generation)
- email reminders for upcoming classes
- german/english language toggle

**medium-term (more complex):**
- exam schedule integration
- building maps with room locations
- professor contact info and office hours
- mobile app (react native)

**long-term (ambitious):**
- personalized recommendations based on interests
- study group formation
- course difficulty ratings from students
- integration with university portal

---

## red flags to avoid

**don't say:**
- ❌ "i don't know how that works"
- ❌ "ai wrote all this code"
- ❌ "i just copied from documentation"
- ❌ "it works on my machine"
- ❌ "i haven't really tested it"

**instead say:**
- ✅ "let me walk through that component..."
- ✅ "i used ai for boilerplate but understand every line"
- ✅ "i researched best practices and adapted them"
- ✅ "i've tested with these specific scenarios..."
- ✅ "i've documented setup and verified on fresh install"

---

## confidence boosters

### you built something real

- functional rag application
- solves actual student problem
- production-ready code quality
- comprehensive documentation
- handles edge cases
- proper error handling
- clean architecture

### you understand the concepts

- can explain rag end-to-end
- understand embeddings and vector search
- know prompt engineering techniques
- grasp intent detection importance
- understand local vs cloud llms

### you made smart decisions

- structured data first (not just rag everything)
- hybrid search (bm25 + vectors)
- proper separation of concerns
- comprehensive logging
- timezone awareness
- conversation history management

---

## last minute prep (1 hour before)

1. **run the app** - verify everything works
2. **test 5 different queries** - schedule, modules, general
3. **review GUIDE.md section on interview questions**
4. **practice explaining the architecture** out loud
5. **review your git commits** - remember what you built when
6. **read through one key file** (chat.py or logic_engine.py)
7. **take a deep breath** - you've got this! 🚀

---

## during the interview

### structure your answers

1. **start high-level** - give the big picture
2. **go deeper if they ask** - provide technical details
3. **connect to their needs** - relate to university use case
4. **be honest** - if you don't know, say how you'd find out

### show enthusiasm

- talk about what excited you while building
- mention what you learned
- express interest in their university's needs
- ask thoughtful questions

### think aloud

when solving problems or answering technical questions:
- verbalize your thought process
- explain why you're considering different approaches
- show how you break down problems

---

## you're ready!

you've built an impressive project. you understand how it works. you can explain the concepts. you've prepared thoroughly.

**trust your preparation. be yourself. good luck!** 🎉
