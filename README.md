# Serene Ease RAG Chatbot

This is a **Retrieval-Augmented Generation (RAG)** system built to provide grounded, source-backed answers to mental health questions using a custom corpus of scraped web data. The system uses **Streamlit** for an interactive chat interface and the **Gemini API** for robust generation.

---

##  Getting Started (Setup and Deployment)

Follow these steps to clone the repository and set up the project on a new machine.

### 1. Prerequisites

Before starting, ensure you have the following installed and configured:

* **Python 3.10+** installed.
* **Git** installed.
* A **Gemini API Key** from Google AI Studio.

### 2. Clone the Repository

Use the `git clone` command to download the project files to your local machine.


# Clone the repository using the HTTPS link
git clone [YOUR_REPOSITORY_LINK_HERE]

# Navigate into the project directory
cd seren_ease_scraper
### 3. Environment Setup
It is highly recommended to use a Python virtual environment to isolate the project dependencies.



### 4. Create a new virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate
# Activation (Windows PowerShell):
# .\venv\Scripts\Activate
4. Install Dependencies
Install all required Python packages using the provided requirements.txt file.


(venv) pip install -r requirements.txt
5. Configure API Key
Set your Gemini API Key as an environment variable. The application will not run without this key.



# IMPORTANT: Replace YOUR_API_KEY_HERE with your actual key
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
 Data Pipeline (One-Time Setup)
The custom knowledge base must be prepared and embedded before the chatbot can function. This step creates the local vector database.

1. Run Data Processing
This step assumes your raw .jsonl data file is present. It cleans the raw data and breaks it into text chunks.



(venv) python data_processing/clean_data.py
2. Run Embedding
This step uses SentenceTransformer to generate vector embeddings and stores them in the chroma_db_serene_ease folder.



(venv) python data_processing/embed_data.py
 Running the Chatbot UI
Once the data pipeline is complete and the API key is set, run the Streamlit application:



(venv) python -m streamlit run app.py
