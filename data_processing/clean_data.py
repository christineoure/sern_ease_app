import pandas as pd
import re
import os

# --- Configuration ---

file_path = "data/mental_health_articles.jsonl" 

# 1. Load the data from the JSON Lines file
try:
    df = pd.read_json(file_path, lines=True)
    print(f"✓ Loaded {len(df)} records from {file_path}.")
except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}. Please check your project structure and file path.")
    # Create an empty DataFrame to prevent immediate crash
    df = pd.DataFrame(columns=['url', 'source', 'title', 'body'])

# 2. Check for missing essential fields and remove those records
initial_count = len(df)
df.dropna(subset=['url', 'title', 'body'], inplace=True)
df = df[df['body'].str.len() > 100] 
print(f"✓ Removed {initial_count - len(df)} records with missing or insufficient content.")

# 3. Deduplicate based on URL (Primary key)
url_duplicates = df.duplicated(subset=['url'], keep='first').sum()
df.drop_duplicates(subset=['url'], keep='first', inplace=True)

# 4. Deduplicate based on Body Content (catching different URLs for same content)
body_duplicates = df.duplicated(subset=['body'], keep='first').sum()
df.drop_duplicates(subset=['body'], keep='first', inplace=True)

print(f"✓ Removed {url_duplicates} URL duplicates.")
print(f"✓ Removed {body_duplicates} Content duplicates.")
print(f"Final record count after Phase 1: {len(df)}")

# 5. Standardize the 'source' domain name
df['source'] = df['source'].str.lower()
df['source'] = df['source'].str.replace('www.', '', regex=False)
df['source'] = df['source'].str.replace('http://', '', regex=False)
df['source'] = df['source'].str.replace('https://', '', regex=False)
df['source'] = df['source'].str.strip()

print("\nSample of cleaned sources:")
print(df['source'].value_counts().head())

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK data is confirmed to be downloaded. Proceeding directly. ---

# Initialize Lemmatizer and Stop Words list
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def advanced_clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    
    # 1. Remove HTML tags, Unicode characters, and any other artifacts
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
    
    # 2. Tokenization: Split text into words
    words = re.findall(r'\b\w+\b', text)
    
    # 3. Lemmatization and Stop Word Removal
    cleaned_words = []
    for word in words:
        if word not in STOP_WORDS:
            cleaned_words.append(lemmatizer.lemmatize(word))
            
    # Rejoin cleaned words into a single string
    return " ".join(cleaned_words)

print("\nStarting Phase 2: Text Preprocessing...")

# Create new columns for the cleaned text
df['clean_title'] = df['title'].apply(advanced_clean_text)
df['clean_body'] = df['body'].apply(advanced_clean_text)

# Final check: Remove any records that became empty after stop word/artifact removal
final_count_before = len(df)
df = df[df['clean_body'].str.len() > 10] 
print(f"✓ Removed {final_count_before - len(df)} records that were reduced to empty text.")

print(f"Final Cleaned DataFrame size: {len(df)}")
print("\n--- Sample of Cleaned Data ---")
print(f"Original Title: {df['title'].iloc[0]}")
print(f"Clean Title:    {df['clean_title'].iloc[0]}")
print(f"Original Body (Start): {df['body'].iloc[0][:150]}...")
print(f"Clean Body (Start):    {df['clean_body'].iloc[0][:150]}...")

# 6. Save the final clean dataset for chunking
df[['url', 'source', 'clean_title', 'clean_body']].to_json('clean_mental_health_articles.jsonl', orient='records', lines=True)
print("\n✓ Cleaned data saved to 'clean_mental_health_articles.jsonl'")

# --- PHASE 3: CHUNKING (Preparing for Embedding/RAG) ---

from nltk.tokenize import sent_tokenize

# Configuration for chunking
CHUNK_SIZE = 500  
CHUNK_OVERLAP = 50 

def create_chunks(row):
    """Splits the clean_body text into overlapping chunks based on sentence boundaries."""
    
    text = row['clean_body']
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        if current_chunk_length + sentence_length > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Overlap logic
            overlap_words = current_chunk[-CHUNK_OVERLAP:]
            current_chunk = overlap_words + sentence_words
            current_chunk_length = len(current_chunk)
            
        else:
            current_chunk.extend(sentence_words)
            current_chunk_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            'url': row['url'],
            'source': row['source'],
            'title': row['clean_title'],
            'chunk_id': f"{row['source']}_{row['url'].split('/')[-1]}_{i}", 
            'chunk_text': chunk
        })
    return chunk_data

# 1. Load the CLEANED data from the new output file
cleaned_file_path = "clean_mental_health_articles.jsonl"
try:
    # Load the file that was just saved in Phase 2
    df_clean = pd.read_json(cleaned_file_path, lines=True)
    print(f"\n✓ Loaded {len(df_clean)} cleaned records for chunking.")
except FileNotFoundError:
    print(f"\nERROR: Cleaned file not found at {cleaned_file_path}. Please check file path.")
    exit()

# 2. Apply the chunking function to every row
print("Applying chunking...")
chunked_list_of_lists = df_clean.apply(create_chunks, axis=1).tolist()

# Flatten the list of lists into a single list of chunk dictionaries
final_chunked_data = [item for sublist in chunked_list_of_lists for item in sublist]

# 3. Create the final DataFrame
df_chunks = pd.DataFrame(final_chunked_data)

# Create a unique article_id and use it to guarantee unique chunk_ids and assign a unique number to each original URL before chunking
df_chunks['article_id'] = df_chunks['url'].astype('category').cat.codes

# re-generate the chunk_id using the guaranteed unique article_id
df_chunks['chunk_id'] = df_chunks.apply(
    lambda row: f"{row['source'].split('.')[0]}_{row['article_id']}_{row.name}",
    axis=1
)

# Save the chunked data
chunked_output_path = 'final_chunked_mental_health_data.jsonl'
df_chunks.to_json(chunked_output_path, orient='records', lines=True)

print(f"\n--- Chunking Results ---")
print(f"Total initial articles: {len(df_clean)}")
print(f"Total chunks created: {len(df_chunks)}")
print(f"✓ Final chunked dataset saved to '{chunked_output_path}'")
print("\nSample Chunk Data:")
print(df_chunks[['chunk_id', 'source', 'chunk_text']].iloc[0])