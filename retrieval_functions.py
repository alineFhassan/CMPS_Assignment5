# retrieval_functions.py

import os
import json
import pickle
import numpy as np
import faiss
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Paths
TRANSCRIPT_PATH = "data/transcript.json"
FAISS_TEXT_INDEX = "retrieval/faiss_text.index"
TFIDF_VECTORIZER = "retrieval/tfidf_vectorizer.pkl"
BM25_MODEL = "retrieval/bm25_model.pkl"

# PostgreSQL Configuration
PG_HOST = "localhost"
PG_PORT = 5432
PG_DBNAME = "rag_db"  
PG_USER = "postgres"
PG_PASSWORD = "password"  

# Model for encoding queries
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load transcript segments
with open(TRANSCRIPT_PATH, "r") as f:
    segments = json.load(f)
texts = [seg['text'] for seg in segments]
timestamps = [seg['start'] for seg in segments]  

# --- Helper functions --- #

def connect_db():
    return psycopg2.connect(
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT
    )

# --- Query Functions --- #

def query_faiss_text(question, top_k=3):
    index = faiss.read_index(FAISS_TEXT_INDEX)
    query_vec = text_encoder.encode([question]).astype(np.float32)
    D, I = index.search(query_vec, top_k)
    results = []
    for idx in I[0]:
        if idx < len(segments):
            results.append({
                "text": segments[idx]['text'],
                "timestamp": segments[idx]['start']
            })
    return results

def query_pgvector(question, method="ivfflat", top_k=3):
    conn = connect_db()
    cursor = conn.cursor()

    query_vec = text_encoder.encode([question])[0]
    embedding_str = "[" + ",".join(map(str, query_vec)) + "]"

    if method == "ivfflat":
        index_name = "ivfflat_index"
    elif method == "hnsw":
        index_name = "hnsw_index"
    else:
        raise ValueError("Invalid method. Choose 'ivfflat' or 'hnsw'.")

    # Search using <-> distance operator
    query = f"""
        SELECT text, id FROM text_embeddings
        ORDER BY embedding <-> '{embedding_str}'
        LIMIT {top_k};
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for text, idx in rows:
        results.append({
            "text": text,
            "timestamp": segments[idx-1]['start'] if idx-1 < len(segments) else 0
        })
    return results

def query_tfidf(question, top_k=3):
    with open(TFIDF_VECTORIZER, "rb") as f:
        vectorizer, tfidf_matrix = pickle.load(f)
    query_vec = vectorizer.transform([question])
    scores = (tfidf_matrix @ query_vec.T).toarray().squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": segments[idx]['text'],
            "timestamp": segments[idx]['start']
        })
    return results

def query_bm25(question, top_k=3):
    with open(BM25_MODEL, "rb") as f:
        bm25 = pickle.load(f)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": segments[idx]['text'],
            "timestamp": segments[idx]['start']
        })
    return results
