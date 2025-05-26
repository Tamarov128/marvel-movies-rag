#!/usr/bin/env python3
"""
Process raw JSON data, chunk into manageable pieces, compute embeddings via a local embedding server
(using gaianet/Nomic-embed-text-v1.5-Embedding-GGUF), and store into a ChromaDB collection.

Assumes fixed paths:
  raw data:    data/raw
  ChromaDB dir: data/chroma
  ChromaDB collection: marvel_films
  Embedding server endpoint: http://127.0.0.1:1234/v1/embeddings
"""
import os
import json
from typing import List

import requests
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# Configuration (hard-coded values)
RAW_DIR = 'data/raw'
CHROMA_DIR = 'data/chroma'
COLLECTION_NAME = 'marvel_films'
EMBED_ENDPOINT = 'http://127.0.0.1:1234/v1/embeddings'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 128  # number of texts per embedding request


class LocalServerEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model: str = "gaianet/Nomic-embed-text-v1.5-Embedding-GGUF"):
        self.endpoint = endpoint
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Send texts to the local embedding server in batches and return embeddings.
        """
        all_embeddings: List[List[float]] = []
        total = len(texts)
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            payload = {"model": self.model, "input": batch}
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            data = result.get("data") or result.get("embeddings") or []
            if len(data) != len(batch):
                raise ValueError(
                    f"Embedding server returned {len(data)} embeddings for {len(batch)} inputs"
                )
            for item in data:
                emb = item.get("embedding") if isinstance(item, dict) else item
                all_embeddings.append(emb)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def process_and_store():
    # 1) Initialize embedding client
    embedder = LocalServerEmbeddings(endpoint=EMBED_ENDPOINT)
    print(f"Using embedding endpoint {EMBED_ENDPOINT} with model '{embedder.model}'...")

    # 2) Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 3) Initialize Persistent Chroma client and collection
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 4) Read and chunk documents
    all_texts = []
    all_metadatas = []
    all_ids = []

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.lower().endswith('.json'):
            continue
        with open(os.path.join(RAW_DIR, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data.get('text', '')
        title = data.get('title', '')
        docs = splitter.create_documents([text], metadatas=[{'source': fname, 'title': title}])
        for i, doc in enumerate(docs):
            all_ids.append(f"{fname}-{i}")
            all_texts.append(doc.page_content)
            all_metadatas.append(doc.metadata)

    # 5) Compute embeddings via local server
    print(f"Requesting embeddings for {len(all_texts)} chunks (batch size = {BATCH_SIZE})...")
    embeddings = embedder.embed_documents(all_texts)

    if not embeddings:
        raise RuntimeError("No embeddings were returned; aborting add to ChromaDB.")

    # 6) Add to Chroma collection
    print(f"Adding embeddings to ChromaDB collection '{COLLECTION_NAME}'...")
    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print("Data successfully added to ChromaDB (persistent storage)")


if __name__ == '__main__':
    process_and_store()
