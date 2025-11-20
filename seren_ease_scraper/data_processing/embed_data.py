# data_processing/embed_data.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os

# --- Configuration ---
CHUNKED_FILE = 'final_chunked_mental_health_data.jsonl'
MODEL_NAME = 'all-MiniLM-L6-v2' 
CHROMA_PATH = 'chroma_db_serene_ease'

def embed_and_store():
    # 1. Load the cleaned and chunked data
    try:
        df = pd.read_json(CHUNKED_FILE, lines=True)
        print(f"✓ Loaded {len(df)} chunks for embedding.")
    except FileNotFoundError:
        print(f"ERROR: Chunked file not found at {CHUNKED_FILE}. Please run clean_data.py first.")
        return

    # 2. Initialize the Embedding Model
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME) 

    # 3. Prepare data for ChromaDB
    
    # Generate guaranteed unique IDs using the DataFrame index
    df['unique_chunk_id'] = df.index.astype(str)
    
    ids = df['unique_chunk_id'].tolist()
    
    # The 'documents' is the text that will be converted to vectors
    documents = df['chunk_text'].tolist()
    
    # Re-add the metadatas definition
    # The 'metadatas' stores the original source information
    metadatas = df[['url', 'source', 'title']].to_dict('records') 
    
    # 4. Initialize ChromaDB Client and Collection
    print(f"Initializing ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create or get a collection. 
    collection_name = f"mental_health_chunks_{MODEL_NAME.split('-')[0]}"
    
    # clear the collection if it already exists to ensure a fresh start
    try:
        client.delete_collection(name=collection_name)
    except:
        pass 
    collection = client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Generate embeddings and add to the database
    print(f"Generating embeddings and adding {len(documents)} documents...")
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"\n--- Embedding and Storage Complete ---")
    print(f"Total chunks embedded: {collection.count()}")
    print(f"The vector database is stored in the '{CHROMA_PATH}' folder.")

if __name__ == "__main__":
    embed_and_store()