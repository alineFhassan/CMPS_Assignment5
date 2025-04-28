# retrieval.py

import os
import json
import numpy as np
import faiss
import pickle
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# PostgreSQL Configuration
PG_HOST = "localhost"
PG_PORT = 5432
PG_DBNAME = "rag_db"
PG_USER = "postgres"
PG_PASSWORD = "password"  

# Paths
TEXT_EMBEDDINGS_PATH = "embeddings/text_embeddings.npy"
IMAGE_EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
TRANSCRIPT_PATH = "data/transcript.json"

# Output Paths
FAISS_TEXT_INDEX = "retrieval/faiss_text.index"
FAISS_IMAGE_INDEX = "retrieval/faiss_image.index"
TFIDF_VECTORIZER = "retrieval/tfidf_vectorizer.pkl"
BM25_MODEL = "retrieval/bm25_model.pkl"

# --- PostgreSQL Functions --- #

def connect_db():
    return psycopg2.connect(
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT
    )

def create_table():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(384)  -- dimension must match your embedding size
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

def insert_text_embeddings(embeddings, texts):
    conn = connect_db()
    cursor = conn.cursor()

    for text, embedding in zip(texts, embeddings):
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        cursor.execute(
            "INSERT INTO text_embeddings (text, embedding) VALUES (%s, %s);",
            (text, embedding_str)
        )

    conn.commit()
    cursor.close()
    conn.close()

def build_pgvector_indexes():
    conn = connect_db()
    cursor = conn.cursor()

    try:
        # First ensure the vector extension is enabled
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create IVFFLAT index with vector_l2_ops operator class
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS ivfflat_index
            ON text_embeddings USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100);  -- Adjust lists based on your dataset size
        """)

        # Create HNSW index with proper parameters
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS hnsw_index
            ON text_embeddings USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64);  -- Good default values
        """)

        conn.commit()
        print("Successfully created PostgreSQL vector indexes")
    except Exception as e:
        conn.rollback()
        print(f"Error creating indexes: {e}")
    finally:
        cursor.close()
        conn.close()

# --- FAISS Functions --- #

def build_faiss_index(embeddings, dim, index_path):
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

# --- Lexical Retrieval Functions (TF-IDF and BM25) --- #

def build_tfidf_model(texts, save_path):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    with open(save_path, "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)

def build_bm25_model(texts, save_path):
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)

# --- Main Script --- #

if __name__ == "__main__":
    os.makedirs("retrieval", exist_ok=True)

    # Load Data
    print("Loading transcript texts...")
    with open(TRANSCRIPT_PATH, "r") as f:
        segments = json.load(f)
    texts = [seg['text'] for seg in segments]

    print("Loading embeddings...")
    text_embeddings = np.load(TEXT_EMBEDDINGS_PATH)
    image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)

    # --- Semantic Retrieval --- #
    print("Building FAISS indexes...")
    build_faiss_index(text_embeddings, text_embeddings.shape[1], FAISS_TEXT_INDEX)
    build_faiss_index(image_embeddings, image_embeddings.shape[1], FAISS_IMAGE_INDEX)

    print("Storing text embeddings into PostgreSQL...")
    create_table()
    insert_text_embeddings(text_embeddings, texts)

    print("Building PostgreSQL IVFFLAT and HNSW indexes...")
    build_pgvector_indexes()

    # --- Lexical Retrieval --- #
    print("Building TF-IDF model...")
    build_tfidf_model(texts, TFIDF_VECTORIZER)

    print("Building BM25 model...")
    build_bm25_model(texts, BM25_MODEL)

    print("Done! Retrieval indexes and models built successfully.")
