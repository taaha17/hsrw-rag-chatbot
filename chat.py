import os
import json
import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from logic_engine import extract_semester_criteria, get_modules_from_map

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
MODULE_MAP_PATH = os.path.join(BASE_DIR, "data", "module_map.json")
EMBEDDING_MODEL = "all-minilm"
CHAT_MODEL = "llama3.2"
OLLAMA_URL = "http://127.0.0.1:11434"

class HttpOllamaEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def _get_embedding(self, text: str):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception:
            return []

    def embed_documents(self, texts):
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text):
        return self._get_embedding(text)

def generate_chat_response(context, question, hardcoded_list=None):
    
    # 1. Dynamic Data Injection
    list_instruction = ""
    if hardcoded_list:
        list_instruction = f"""
        [SYSTEM NOTICE]: The user's question implies a need for a list of modules. 
        I have retrieved the OFFICIAL data:
        {hardcoded_list}
        
        INSTRUCTION: Present this list immediately. Do not say "I cannot find it". 
        After the list, you may add details from the context if relevant.
        """

    # 2. Global Static Knowledge (The "Common Sense" layer)
    global_knowledge = """
    DEGREE FACTS:
    - Name: Infotronic Systems Engineering (ISE), formerly Communication and Information Engineering (CIE).
    - Duration: 7 Semesters (Standard Period of Study).
    - Total Credits: 210 ECTS.
    - Structure: Semesters 1-3 (Basics), 4-5 (Advanced/Electives), 6 (Internship/Abroad), 7 (Thesis).
    """

    system_prompt = f"""
    You are the friendly AI Advisor for the "Infotronic Systems Engineering" (ISE) degree at Rhine-Waal University.
    
    {global_knowledge}
    
    {list_instruction}
    
    GUIDELINES:
    1. CONVERSATIONAL: Be helpful and direct. If asked "What do I study?", assume they want the module list for their current semester.
    2. NO BUREAUCRACY: Do not mention "Credit Points" or "Admission Rules" unless the student explicitly asks about "requirements" or "eligibility". Assume the student is qualified.
    3. ACCURACY: If the [SYSTEM NOTICE] provides a list, use it as the absolute truth.
    
    Context from Documents:
    {context}
    
    Student Question: {question}
    """
    
    data = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": system_prompt}],
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=data, timeout=60)
        return response.json()["message"]["content"]
    except Exception as e:
        return f"❌ Chat Error: {e}"

def chat_loop():
    print(f"🤖 Loading Brain...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Error: DB not found.")
        return

    module_map = {}
    if os.path.exists(MODULE_MAP_PATH):
        with open(MODULE_MAP_PATH, "r", encoding="utf-8") as f:
            module_map = json.load(f)

    # Initialize Hybrid Search
    embedding_function = HttpOllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    all_docs = vector_store.get()["documents"]
    metadatas = vector_store.get()["metadatas"]
    doc_objects = [Document(page_content=t, metadata=m) for t, m in zip(all_docs, metadatas)]
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.4, 0.6]
    )
    
    print(f"✅ Ready! (Indexed {len(module_map)} modules)")
    print("------------------------------------------------")

    while True:
        query_text = input("\nYou: ")
        if query_text.lower() in ["exit", "quit"]:
            break
            
        print("🔍 Searching...", end="", flush=True)
        
        # 1. Logic Check (Semester Extraction)
        criteria = extract_semester_criteria(query_text)
        hardcoded_list = None
        
        if criteria["semester_num"] or criteria["season"]:
            raw_list = get_modules_from_map(module_map, criteria)
            if raw_list:
                hardcoded_list = "\n".join(raw_list)
                print(f" (Logic Engine found {len(raw_list)} modules)")

        # 2. Context Retrieval
        results = ensemble_retriever.invoke(query_text)
        context_text = "\n\n".join([f"--- SOURCE: {doc.metadata.get('source')} ---\n{doc.page_content}" for doc in results])
        
        print("\r🤖 Thinking... ", end="", flush=True)
        answer = generate_chat_response(context_text, query_text, hardcoded_list)
        
        print(f"\rBot: {answer}\n")

if __name__ == "__main__":
    chat_loop()