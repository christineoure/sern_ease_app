# app.py

import streamlit as st
import chromadb
from google import genai
import os

# --- Configuration  ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
CHROMA_PATH = 'chroma_db_serene_ease'
COLLECTION_NAME = f"mental_health_chunks_{MODEL_NAME.split('-')[0]}"
N_RESULTS = 3 # Number of relevant chunks to retrieve
GEMINI_MODEL = "gemini-2.5-flash" 

# --- Function to initialize the RAG backend ---

@st.cache_resource
def get_rag_components():
    """Initializes ChromaDB client and Gemini client once."""
    
    # Check for API Key
    if not os.getenv("GEMINI_API_KEY"):
        st.error("FATAL ERROR: GEMINI_API_KEY environment variable not set. Please set it to run the app.")
        return None, None
    
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        # Initialize Gemini Client
        gemini_client = genai.Client()
        
        return collection, gemini_client
    except Exception as e:
        st.error(f"Error initializing RAG components (ChromaDB or Gemini): {e}")
        return None, None

# --- Main RAG Query Logic ---

def run_rag_query(user_query, collection, gemini_client):
    """Performs the full RAG process and returns the generated text and sources."""
    
    # 1. RETRIEVAL (R)
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
        
        context_snippets.append(f"Source {i+1} ({source}): {document}")
        sources.add(f"**[{source}]**: {url}")
        
    context_text = "\n\n".join(context_snippets)
    
    # 2. PROMPT CONSTRUCTION
    system_instruction = (
        "You are an expert mental health summarization assistant. "
        "Your task is to synthesize a coherent and specific answer to the user's question "
        "based **ONLY** on the context provided in the snippets. "
        "If the snippets do not contain the information, state that clearly."
    )
    
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
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=rag_prompt,
        config=dict(
            system_instruction=system_instruction
        )
    )
    
    # Format the answer to include sources nicely
    source_list = "\n".join(sorted(list(sources)))
    full_response = (
        f"{response.text}\n\n"
        f"--- \n"
        f"**Sources Used:**\n{source_list}"
    )
    
    return full_response

# --- Streamlit UI Components ---

def main():
    st.set_page_config(page_title="Serene Ease RAG Chatbot", layout="wide")
    st.title("Serene Ease: Mental Health RAG Chatbot")
    st.markdown("Ask a question about mental health, and the system will retrieve relevant context from your custom knowledge base and use Gemini to generate a grounded answer.")
    
    # Initialize RAG components
    collection, gemini_client = get_rag_components()
    
    if collection is None or gemini_client is None:
        return # Stop if initialization failed

    # --- BLOCK ADDED TO MAINTAIN CHAT HISTORY ---
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_query := st.chat_input("Ask a question about mental health..."):
        
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get and display the RAG response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    # Run the RAG pipeline
                    full_response = run_rag_query(user_query, collection, gemini_client)
                    
                    st.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_message = f"An error occurred: {e}. Please check your API key and data paths."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # --- END OF CHAT HISTORY BLOCK ---

if __name__ == "__main__":
    main()