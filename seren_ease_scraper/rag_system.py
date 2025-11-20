# rag_system.py

import chromadb
from google import genai
import os

# --- Configuration ---
# Retrieval Settings
MODEL_NAME = 'all-MiniLM-L6-v2' 
CHROMA_PATH = 'chroma_db_serene_ease'
COLLECTION_NAME = f"mental_health_chunks_{MODEL_NAME.split('-')[0]}"
N_RESULTS = 3 # Number of relevant chunks to retrieve

# Generation Settings
GEMINI_MODEL = "gemini-2.5-flash" 

def run_rag_query(user_query: str):
    """
    Performs the full RAG process: Retrieves context from the vector DB,
    then uses Gemini to generate a final answer based on that context.
    """
    # 1. RETRIEVAL (R)
    print("--- 1. RETRIEVAL (Searching Vector DB) ---")
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Query the database
        results = collection.query(
            query_texts=[user_query],
            n_results=N_RESULTS,
            include=['documents', 'metadatas']
        )
        
        # Compile retrieved snippets and their sources
        context_snippets = []
        sources = set()

        for i in range(N_RESULTS):
            document = results['documents'][0][i]
            source = results['metadatas'][0][i]['source']
            url = results['metadatas'][0][i]['url']
            
            # Format snippet for the prompt
            context_snippets.append(f"Source {i+1} ({source}): {document}")
            sources.add(f"[{source}]: {url}")
            
        context_text = "\n\n".join(context_snippets)
        print(f"✓ Retrieved {len(context_snippets)} relevant chunks.")

    except Exception as e:
        print(f"\nERROR during Retrieval: {e}")
        return

    # 2. PROMPT CONSTRUCTION
    
    # Define a clear system instruction to guide the LLM's behavior
    system_instruction = (
        "You are an expert mental health summarization assistant. "
        "Your task is to synthesize a coherent and specific answer to the user's question "
        "based **ONLY** on the context provided in the snippets. "
        "If the snippets do not contain the information, state that clearly."
    )
    
    # Construct the final prompt containing the instructions, context, and query
    rag_prompt = f"""
    CONTEXT SNIPPETS:
    ---
    {context_text}
    ---
    
    USER QUESTION:
    {user_query}
    
    Synthesize an answer using only the provided CONTEXT SNIPPETS.
    """
    
    # 3. GENERATION (G)
    print("\n--- 2. GENERATION (Calling Gemini) ---")
    
    try:
        # Initialize the Gemini client
        gemini_client = genai.Client()

        # Call the Gemini API
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=rag_prompt,
            config=dict(
                system_instruction=system_instruction
            )
        )
        
        # 4. Output Final Answer
        print("\n=======================================================")
        print(f" GENERATED ANSWER ({GEMINI_MODEL})")
        print("=======================================================")
        print(response.text)
        
        print("\n--- SOURCES USED ---")
        for source in sorted(list(sources)):
            print(source)

    except Exception as e:
        print(f"\nERROR during Generation: Could not connect to Gemini API. Ensure GEMINI_API_KEY is set correctly. Details: {e}")


if __name__ == "__main__":
    # Example Query 
    rag_query = "What are practical, daily methods for stress management and preventing anxiety according to the data?"
    run_rag_query(rag_query)