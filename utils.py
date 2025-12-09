"""
Utility Functions - Custom Ollama Embeddings Implementation

This module provides a custom embeddings class for Ollama that uses HTTP requests
instead of the official Ollama Python client. This gives us more control and
better error handling for the RAG pipeline.

Why custom implementation?
- Better timeout handling for large documents
- Explicit error messages for debugging
- Compatible with LangChain's Embeddings interface
- Can be easily modified for batch processing improvements
"""

import requests
from langchain_core.embeddings import Embeddings

# --- CONFIGURATION ---
# Note: This URL should match the OLLAMA_URL in config.py
# Keeping it here for module independence, but consider importing from config in production
OLLAMA_URL = "http://127.0.0.1:11434"

class HttpOllamaEmbeddings(Embeddings):
    """
    Custom embeddings class for Ollama using HTTP REST API.
    
    This class implements LangChain's Embeddings interface to generate
    vector embeddings for text using a locally-hosted Ollama model.
    
    Usage:
        embeddings = HttpOllamaEmbeddings(model="all-minilm")
        vectors = embeddings.embed_documents(["text1", "text2"])
        query_vector = embeddings.embed_query("search query")
    """
    def __init__(self, model: str):
        self.model = model

    def _get_embedding(self, text: str):
        """
        Generates an embedding for a single piece of text.
        """
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP Request failed: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            return []

    def embed_documents(self, texts):
        """
        Generates embeddings for a list of documents.
        """
        # TODO: Investigate batch embedding with Ollama for better performance.
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text):
        """
        Generates an embedding for a single query.
        """
        return self._get_embedding(text)
