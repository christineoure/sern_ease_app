import streamlit as st
import chromadb
from google import genai
import os

# --- 1. Configuration (Synced with rag_system.py & query_db.py) ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
# Using absolute paths ensures Streamlit Cloud finds the folder
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(ABS_PATH, 'chroma_db_serene_ease')

# This must match your backend files exactly
COLLECTION_NAME = f"mental_health_chunks_{MODEL_NAME.split('-')[0]}"
N_RESULTS = 2 
GEMINI_MODEL = "gemini-2.0-flash" # Use 2.0-flash for stability

# --- 2. Backend Initialization ---

@st.cache_resource
def get_rag_components():
    """Initializes ChromaDB and Gemini client with Cloud-safe paths."""
    
    # Check for API Key in Streamlit Secrets
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY. Please add it to Secrets.")
        return None, None

    # Sidebar Diagnostics (Helps us verify the push worked)
    st.sidebar.subheader("System Status")
    if os.path.exists(CHROMA_PATH):
        st.sidebar.success("Knowledge Base Found")
        st.sidebar.write(f"Files: {os.listdir(CHROMA_PATH)}")
    else:
        st.sidebar.error("Knowledge Base Missing")
        st.sidebar.info("Run 'git add -f chroma_db_serene_ease/chroma.sqlite3' locally.")

    try:
        # Connect to the existing DB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Verify collection exists
        existing_cols = [c.name for c in chroma_client.list_collections()]
        if COLLECTION_NAME not in existing_cols:
            st.sidebar.warning(f"Collection '{COLLECTION_NAME}' not found. Available: {existing_cols}")
            # Fallback to the first available if possible
            if existing_cols:
                target = existing_cols[0]
            else:
                return None, None
        else:
            target = COLLECTION_NAME

        collection = chroma_client.get_collection(name=target)
        gemini_client = genai.Client(api_key=api_key)
        
        st.sidebar.write(f"Active Collection: `{target}`")
        return collection, gemini_client

    except Exception as e:
        st.sidebar.error(f"Init Error: {e}")
        return None, None

# --- 3. RAG Logic (Synced with rag_system.py) ---

def run_rag_query(user_query, collection, gemini_client):
    """Retrieves context and generates response."""
    
    # RETRIEVAL (R)
    results = collection.query(
        query_texts=[user_query],
        n_results=N_RESULTS,
        include=['documents', 'metadatas']
    )
    
    context_snippets = []
    sources = set()

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        src = meta.get('source', 'Unknown')
        url = meta.get('url', '#')
        
        context_snippets.append(f"Source {i+1} ({src}): {doc}")
        sources.add(f"**[{src}]**: {url}")
        
    context_text = "\n\n".join(context_snippets)
    
    # GENERATION (G)
    system_instruction = (
        "You are an expert mental health summarization assistant. "
        "Answer based ONLY on the provided snippets. If unsure, say so."
    )
    
    rag_prompt = f"CONTEXT:\n{context_text}\n\nUSER QUESTION:\n{user_query}"
    
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=rag_prompt,
        config=dict(system_instruction=system_instruction)
    )
    
    source_list = "\n".join(sorted(list(sources)))
    return f"{response.text}\n\n---\n**Sources Used:**\n{source_list}"

# --- 4. Streamlit UI ---

def main():
    st.set_page_config(page_title="Serene Ease", page_icon="🌿")
    st.title("🌿 Serene Ease: Mental Health AI")
    st.caption("Grounded in verified mental health resources.")

    collection, gemini_client = get_rag_components()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if collection and (user_input := st.chat_input("How can I help you today?")):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching resources..."):
                answer = run_rag_query(user_input, collection, gemini_client)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
