import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. SET YOUR API KEY
#os.environ["GOOGLE_API_KEY"] = "AIzaSyAl5iq8WX3Ga04RqT8SkXAsMD77BN1Cevs"
api_key = os.getenv("GOOGLE_API_KEY")

def initialize_knowledge_base():
    initial_knowledge = [
        "Project Risk Policy: All high-complexity projects require weekly audits.",
        "Clinical Data Safety: Cardiac scan datasets must be stored in encrypted S3 buckets.",
        "Team Turnover: Rates above 15% trigger an automatic risk escalation.",
        "Financial Compliance: Transaction defaults over $50,000 must be reported.",
        "Market Strategy: Inflation above 5% requires a 10% budget reallocation."
    ]

    print("🔄 Generating 'chroma_db' folder... please wait.")

    try:
        # 3. Use the stable model with an explicit API version
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
        
        #embeddings = GoogleGenerativeAIEmbeddings(
            #model="models/text-embedding-004",
            model_api_version="v1"  # This is the key fix for the 404 error
        )
        

        # 4. Create the Vector Database
        vector_db = Chroma.from_texts(
            texts=initial_knowledge,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="risk_policies"
        )
        print("✅ Success! The 'chroma_db' folder has been created.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    initialize_knowledge_base()