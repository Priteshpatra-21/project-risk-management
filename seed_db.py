import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Setup API Key from your existing secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 2. Define the knowledge you want to start with
initial_knowledge = [
    "Project Risk Policy: All high-complexity projects require weekly audits.",
    "Clinical Data Safety: All cardiac scans must be anonymized before S3 upload.",
    "Company Protocol: Transaction defaults over $10k must be flagged."
]

# 3. Create the database (This generates the 'chroma_db' folder)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = Chroma.from_texts(
    texts=initial_knowledge,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="risk_policies"
)

print("✅ Success! The 'chroma_db' folder has been created locally.")
