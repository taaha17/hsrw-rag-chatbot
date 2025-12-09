import os
import shutil
import json
import re
import requests
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
MODULE_MAP_PATH = os.path.join(DATA_FOLDER, "module_map.json")
OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
EMBEDDING_MODEL = "all-minilm"

# Regex: Matches "CI_X.YY" followed by text
# Examples: "CI_1.02 Fundamentals", "CI_W.01 Ambient"
MODULE_PATTERN = re.compile(r"^(CI_[W|K|\d]\.\d{2})\s+(.+)$")

class HttpOllamaEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def _get_embedding(self, text: str):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": text},
                timeout=120
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception:
            return []

    def embed_documents(self, texts):
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text):
        return self._get_embedding(text)

def is_valid_name(name):
    # 1. Reject if it contains technical stats
    bad_keywords = ["150 h", "300 h", "CP", "semester", "Workload", "Duration", "Code"]
    for word in bad_keywords:
        if word in name:
            return False
            
    # 2. Reject if too short
    if len(name) < 5:
        return False
        
    # 3. NEW: Reject if it looks like a description sentence
    # (Real titles don't start with lowercase or quotes)
    if name.strip().startswith('"') or name.strip()[0].islower():
        return False
        
    return True

def semantic_chunking(file_path, filename):
    loader = PDFPlumberLoader(file_path)
    raw_docs = loader.load()
    
    full_text = "\n".join([doc.page_content for doc in raw_docs])
    lines = full_text.split('\n')
    
    chunks = []
    module_map = {}
    
    current_code = None
    current_title = None
    current_buffer = []
    
    print(f"   - Semantically parsing {filename}...")

    for line in lines:
        line = line.strip()
        if not line: continue

        match = MODULE_PATTERN.match(line)
        
        # Found a potential Module Header?
        if match:
            code = match.group(1)
            raw_name = match.group(2).strip()
            
            # CRITICAL CHECK: Is this actually a title, or just table data?
            if is_valid_name(raw_name):
                
                # 1. Save Previous Module
                if current_code:
                    chunks.append(Document(
                        page_content="\n".join(current_buffer),
                        metadata={"source": filename, "code": current_code, "title": current_title}
                    ))
                
                # 2. Start New Module
                current_code = code
                current_title = raw_name
                current_buffer = [line]
                
                # Update Dictionary (Only if valid!)
                # We use setdefault to ensure the FIRST find (usually the header) isn't overwritten
                if current_code not in module_map:
                    module_map[current_code] = current_title
                    print(f"     ✅ Found: {current_code} - {current_title}")
                
            else:
                # It matched regex (CI_X.YY) but failed validation (was table data)
                # Just add it to the buffer of the current module
                if current_code:
                    current_buffer.append(line)
        else:
            # Normal text line
            if current_code:
                current_buffer.append(line)

    # Save the very last module
    if current_code:
        chunks.append(Document(
            page_content="\n".join(current_buffer),
            metadata={"source": filename, "code": current_code, "title": current_title}
        ))

    return chunks, module_map

def ingest_documents():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"🧹 Cleared database.")

    all_chunks = []
    master_module_map = {}

    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder '{DATA_FOLDER}' not found.")
        return

    for filename in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if filename.endswith(".pdf") and "MH" in filename:
            chunks, mod_map = semantic_chunking(file_path, filename)
            all_chunks.extend(chunks)
            master_module_map.update(mod_map)
            
        elif filename.endswith(".pdf"):
            print(f"   - Standard parsing {filename}...")
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            for doc in split_docs:
                doc.metadata["source"] = filename
            all_chunks.extend(split_docs)

    # Save Clean Map
    with open(MODULE_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(master_module_map, f, indent=4)
    print(f"✅ Saved {len(master_module_map)} clean modules to map.")

    if all_chunks:
        print(f"🧠 Embedding {len(all_chunks)} chunks...")
        embedding_model = HttpOllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )
        print(f"✅ Database created.")

if __name__ == "__main__":
    ingest_documents()