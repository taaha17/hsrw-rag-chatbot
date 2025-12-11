# quick reference card - keep this open during interview

## architecture in 30 seconds
```
user query
  → detect intent (schedule/modules/info)
  → get data (json or rag search)
  → build context
  → send to llm (ollama/llama3.2)
  → return response
```

## key files - what they do
- **app.py** - gradio web ui, event handlers
- **chat.py** - llm communication, prompt engineering
- **logic_engine.py** - intent detection, data filtering
- **ingest.py** - pdf parsing, vector db creation
- **config.py** - settings, paths, logging
- **utils.py** - custom ollama embeddings
- **run.py** - auto-detect and launch

## tech stack - why chosen
- **python 3.11** - modern, great libraries
- **ollama** - local llm, private, no api costs
- **langchain** - rag framework
- **chromadb** - vector database
- **gradio** - web ui in pure python
- **llama3.2** - 7b chat model
- **all-minilm** - lightweight embeddings

## rag explained
1. chunk documents → 2. embed → 3. store in chromadb
4. query comes in → 5. embed query → 6. similarity search
7. retrieve top k chunks → 8. inject into prompt → 9. llm generates answer

## why this approach?
- **structured data first** - json for facts (schedules)
- **rag for details** - search pdfs for descriptions
- **prevents hallucination** - grounded in real data
- **local llm** - privacy, control, no costs
- **hybrid search** - bm25 (keywords) + vectors (semantics)

## critical bug fixed
**problem:** gradio 6.x uses multimodal format: `[{'text': '...', 'type': 'text'}]`
**issue:** ollama expects plain strings, not lists
**solution:** `extract_text_from_message()` function converts format
**result:** multi-turn conversations now work perfectly

## demo flow
1. launch: `python run.py`
2. ask schedule: "when is signals class?"
3. ask module: "tell me about physics"
4. show follow-up works: "who teaches it?"
5. show logs: open logs/zero_chat.log

## if asked to improve
**quick wins:** more programs, calendar export, language toggle
**medium:** exam schedules, building maps, mobile app
**ambitious:** personalized recommendations, study groups, portal integration

## if asked about scale
- redis caching for common queries
- load balancer with multiple instances
- separate vector db read replicas
- query queue with workers
- cdn for static assets

## key numbers
- 37 modules extracted
- 21 schedule entries
- 67 document chunks
- 384 embedding dimensions
- 7b parameter model (llama3.2)
- 3 log files (chat, debug, prompts)

## confident statements
✅ "i fixed the gradio 6.x multimodal format bug"
✅ "i implemented hybrid search for better retrieval"
✅ "i prioritize structured data over llm guessing"
✅ "i built comprehensive logging for debugging"
✅ "i understand the entire rag pipeline"

## honest responses
if unsure: "let me think through this..." then explain reasoning
if don't know: "i'm not familiar with that, but i'd research..."
show process: "first i'd check... then i'd try... finally..."

## remember
- start high-level, go deeper if asked
- connect to their university needs
- show enthusiasm for what you learned
- think aloud when problem-solving
- be yourself - you've got this! 🚀
