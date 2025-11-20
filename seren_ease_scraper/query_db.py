# query_db.py

import chromadb

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
CHROMA_PATH = 'chroma_db_serene_ease'
COLLECTION_NAME = f"mental_health_chunks_{MODEL_NAME.split('-')[0]}"

def query_vector_db(query_text: str, n_results: int = 3):
    """
    Connects to the ChromaDB, queries the collection with the provided text,
    and returns the top N most relevant chunks.
    """
    try:
        # 1. Initialize ChromaDB Client
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        print(f"Searching database for: **'{query_text}'**")
        
        # 2. Perform the Query
        # Chroma automatically uses the embedding model defined during the 'add' process
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # 3. Format and Display Results
        print("\n--- Top Retrieved Contexts ---")
        
        # Process the results dictionary
        for i in range(n_results):
            source = results['metadatas'][0][i]['source']
            url = results['metadatas'][0][i]['url']
            document = results['documents'][0][i]
            distance = results['distances'][0][i]
            
            print(f"\n#️⃣ Result {i+1} (Similarity Distance: {distance:.4f})")
            print(f"Source: **{source}**")
            print(f"URL: {url}")
            print(f"Snippet: *{document[:200]}...*") # Print the first 200 characters

    except ValueError as e:
        print(f"\nERROR: Could not find collection '{COLLECTION_NAME}' or database at '{CHROMA_PATH}'.")
        print("Please ensure embed_data.py ran successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Example query: Ask a mental health question that should retrieve relevant documents
    user_query = "what are the most important ways to prevent anxiety in daily life"
    query_vector_db(user_query, n_results=3)