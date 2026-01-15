# app.py

import streamlit as st
import chromadb
from google import genai
import os

# --- Configuration  ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
CHROMA_PATH = 'chroma_db_serene_ease'
COLLECTION_NAME = f"mental_health_chunks_{MODEL_NAME.split('-')[0]}"
N_RESULTS = 3 
GEMINI_MODEL = "gemini-2.0-flash" 

# --- Function to initialize the RAG backend ---

@st.cache_resource
def get_rag_components():
    """Initializes ChromaDB client and Gemini client once with detailed debugging."""
    
    # 1. Look for API Key
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("FATAL ERROR: GEMINI_API_KEY not found. Please set it in Streamlit Cloud Secrets.")
        return None, None
    
    # --- DEBUG SECTION ---
    st.info("System Debug Mode")
    st.write(f"Current Working Directory: `{os.getcwd()}`")
    
    # Check if the folder exists at all
    if os.path.exists(CHROMA_PATH):
        st.success(f"Folder `{CHROMA_PATH}` exists.")
        st.write(f"Folder contents: {os.listdir(CHROMA_PATH)}")
    else:
        st.error(f"Folder `{CHROMA_PATH}` NOT found!")
        st.write(f"Top-level files in GitHub repo: {os.listdir('.')}")
    # ----------------------
    
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # 2. Check available collections
        existing_collections = [c.name for c in chroma_client.list_collections()]
        st.write(f"Available Collections: {existing_collections}")
        
        if COLLECTION_NAME not in existing_collections:
            st.warning(f"Collection '{COLLECTION_NAME}' missing. Expected one of: {existing_collections}")
            return None, None
            
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        # 3. Initialize Gemini Client
        gemini_client = genai.Client(api_key=api_key)
        
        return collection, gemini_client
        
    except Exception as e:
        st.error(f"Error during RAG initialization: {e}")
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
    
    context_snippets = []
    sources = set()

    for i in range(len(results['documents'][0])):
        document = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        source = meta.get('source', 'Unknown')
        url = meta.get('url', '#')
        
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
    st.markdown("Retrieving context from your custom knowledge base to generate grounded answers.")
    
    collection, gemini_client = get_rag_components()
    
    if collection is None or gemini_client is None:
        st.warning("Awaiting proper database initialization...")
        return 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about mental health..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    full_response = run_rag_query(user_query, collection, gemini_client)
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Generation Error: {e}")

if __name__ == "__main__":
    main()