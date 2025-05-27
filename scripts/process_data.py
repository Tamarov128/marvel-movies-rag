#!/usr/bin/env python3
"""
Process raw JSON data, chunk into manageable pieces with RecursiveCharacterTextSplitter,
compute embeddings via a local embedding server, and store into a ChromaDB collection.
Paths are resolved relative to the project root (one level up from this script).
"""
import os
import json
import requests
from typing import List

import chromadb
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ─── Project-root-aware paths ─────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw", "raw_marvel_wiki")
# CHROMA_DIR   = os.path.join(PROJECT_ROOT, "data", "chroma", "chroma_marvel_wiki")

# RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw", "raw_wikipedia")
# CHROMA_DIR   = os.path.join(PROJECT_ROOT, "data", "chroma", "chroma_wikipedia")

RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw", "raw_mcu_wiki")
CHROMA_DIR   = os.path.join(PROJECT_ROOT, "data", "chroma", "chroma_mcu_wiki")

# ─── Configuration ─────────────────────────────────────────────────────────────
COLLECTION_NAME = "marvel_films"
EMBED_ENDPOINT  = "http://127.0.0.1:1234/v1/embeddings"
CHUNK_SIZE      = 1000   # characters per chunk
CHUNK_OVERLAP   = 200    # overlap between chunks
BATCH_SIZE      = 128    # number of texts per embedding request

class LocalServerEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model: str = "gaianet/Nomic-embed-text-v1.5-Embedding-GGUF"):
        self.endpoint = endpoint
        self.model    = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                resp = requests.post(self.endpoint, json={"model": self.model, "input": batch})
                resp.raise_for_status()
                result = resp.json()
                data = result.get("data") or result.get("embeddings") or []
                if len(data) != len(batch):
                    raise ValueError(f"Expected {len(batch)} embeddings, got {len(data)}")
                for item in data:
                    emb = item.get("embedding") if isinstance(item, dict) else item
                    embeddings.append(emb)
            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE + 1}: {e}")
                raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def main():
    # 1) Ensure directories exist
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 2) Load & split raw JSON documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs, metadatas, ids = [], [], []

    json_files = [f for f in sorted(os.listdir(RAW_DIR)) if f.lower().endswith('.json')]
    print(f"Processing {len(json_files)} JSON files...")

    for fname in json_files:
        path = os.path.join(RAW_DIR, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text  = data.get('text', '')
            title = data.get('title', '')
            
            if not text.strip():
                print(f"Warning: No text content in {fname}")
                continue
                
            chunks = splitter.create_documents(
                [text],
                metadatas=[{'source': fname, 'title': title}]
            )
            
            for i, chunk in enumerate(chunks):
                docs.append(chunk.page_content)
                metadatas.append(chunk.metadata)
                ids.append(f"{fname}-{i}")
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    if not docs:
        print("No documents to process!")
        return

    print(f"Created {len(docs)} chunks from {len(json_files)} files")

    # 3) Initialize ChromaDB client with persistent storage
    try:
        # Create ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Delete existing collection if it exists (for clean restart)
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass  # Collection doesn't exist, which is fine
        
        # Create embedder and vector store
        embedder = LocalServerEmbeddings(endpoint=EMBED_ENDPOINT)
        vectordb = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=CHROMA_DIR
        )
        
        print(f"Adding {len(docs)} chunks to ChromaDB...")
        
        # Add documents in batches to avoid memory issues
        batch_size = 50  # Smaller batch size for adding to avoid issues
        for i in range(0, len(docs), batch_size):
            end_idx = min(i + batch_size, len(docs))
            batch_docs = docs[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            vectordb.add_texts(
                texts=batch_docs, 
                metadatas=batch_metadatas, 
                ids=batch_ids
            )
        
        # Verify the data was added
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Ingestion complete! Collection '{COLLECTION_NAME}' contains {count} documents")
        
    except Exception as e:
        print(f"Error during ChromaDB operations: {e}")
        raise

if __name__ == '__main__':
    main()