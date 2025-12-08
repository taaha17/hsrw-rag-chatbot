import os
import shutil
import requests
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# --- CONFIGURATION ---
DATA_FOLDER = "./data"
DB_PATH = "./chroma_db"
OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"

# --- CUSTOM "DIRECT" CONNECTION CLASS ---
# This bypasses the 'ollama' library and talks directly to the server
class HttpOllamaEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def _get_embedding(self, text: str) -> List[float]:
        try:
            # We use standard 'requests' which we know works on your machine
            response = requests.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": text},
                timeout=120  # Wait up to 2 mins for a response
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"❌ Error talking to Ollama: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process one by one (simple and robust)
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

def ingest_documents():
    # 1. Clear old database
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"🧹 Cleared database at {DB_PATH}")
        except Exception as e:
            print(f"⚠️ Could not delete old DB (file in use?): {e}")

    # 2. Load PDFs
    documents = []
    print("📂 Loading PDFs...")
    
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder '{DATA_FOLDER}' not found.")
        return

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_FOLDER, filename)
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            print(f"   - Loaded {filename}: {len(loaded_docs)} pages")
            documents.extend(loaded_docs)

    if not documents:
        print("⚠️ No PDFs found in 'data' folder.")
        return

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split {len(documents)} pages into {len(chunks)} chunks.")

    # 4. Embed & Store
    print("🧠 Embedding... (Using Direct HTTP Connection)")
    
    # Use our new class that uses 'requests'
    embedding_model = HttpOllamaEmbeddings(model="all-minilm")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"✅ Success! Database created at '{DB_PATH}'")

if __name__ == "__main__":
    ingest_documents()