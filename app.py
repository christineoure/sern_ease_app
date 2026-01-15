# app.py

import streamlit as st
import chromadb
from google import genai
import os

# --- Configuration  ---
# 1. Use absolute pathing to ensure the Cloud finds the folder regardless of where it starts
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(ABS_PATH, 'chroma_db_serene_ease')

# 2. Hardcode the collection name to exactly what you created locally 
# (Based on your error, it was 'mental_health_chunks_all')
COLLECTION_NAME = "mental_health_chunks_all" 

N_RESULTS = 3 
GEMINI_MODEL = "gemini-2.0-flash" 

# --- Function to initialize the RAG backend ---

@st.cache_resource
def get_rag_components():
    """Initializes ChromaDB and Gemini with a Collection Inspector."""
    
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("FATAL ERROR: GEMINI_API_KEY missing in Streamlit Secrets.")
        return None, None
    
    # 1. Setup Absolute Path (Ensures the cloud finds the folder)
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    CHROMA_PATH = os.path.join(ABS_PATH, 'chroma_db_serene_ease')
    
    try:
        # 2. Connect to Chroma
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # 3. INSPECTOR: List what is actually inside the DB
        all_collections = chroma_client.list_collections()
        existing_names = [c.name for c in all_collections]
        
        # Display this to you in the app so you can see the truth
        if not existing_names:
            st.error(f"📂 Database connected, but it is EMPTY. No collections found in `{CHROMA_PATH}`.")
            st.write("Check if your `chroma.sqlite3` file was actually pushed to GitHub.")
            return None, None
            
        st.info(f"📂 Database connected. Found collections: `{existing_names}`")

        # 4. Attempt to grab the collection
        # This MUST match one of the names in 'existing_names'
        TARGET_NAME = "mental_health_chunks_all" 
        
        if TARGET_NAME not in existing_names:
            st.warning(f"⚠️ Looking for '{TARGET_NAME}', but found {existing_collections}. Using '{existing_names[0]}' instead.")
            TARGET_NAME = existing_names[0] # Auto-select the first one found

        collection = chroma_client.get_collection(name=TARGET_NAME)
        gemini_client = genai.Client(api_key=api_key)
        
        return collection, gemini_client
        
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

# --- Main RAG Query Logic ---

def run_rag_query(user_query, collection, gemini_client):
    """Performs the full RAG process and returns the generated text and sources."""
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
    
    system_instruction = (
        "You are an expert mental health summarization assistant. "
        "Your task is to synthesize a coherent and specific answer to the user's question "
        "based **ONLY** on the context provided in the snippets. "
        "If the snippets do not contain the information, state that clearly."
    )
    
    rag_prompt = f"CONTEXT SNIPPETS:\n{context_text}\n\nUSER QUESTION:\n{user_query}"
    
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=rag_prompt,
        config=dict(system_instruction=system_instruction)
    )
    
    source_list = "\n".join(sorted(list(sources)))
    return f"{response.text}\n\n---\n**Sources Used:**\n{source_list}"

# --- Streamlit UI Components ---

def main():
    st.set_page_config(page_title="Serene Ease RAG Chatbot", layout="wide")
    st.title("Serene Ease: Mental Health RAG Chatbot")
    
    collection, gemini_client = get_rag_components()
    
    if collection is None or gemini_client is None:
        st.warning("Awaiting database connection...")
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