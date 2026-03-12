import os
import shutil
import boto3
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from typing import Annotated, TypedDict, List

# Core Agentic & RAG Imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. LIGHT-THEME UI SETUP ---
st.set_page_config(page_title="Risk Intel Pro", layout="wide", page_icon="🛡️")

# Constants for S3 & Vector DB
LOCAL_DB_PATH = "./chroma_db"
S3_VECTOR_PREFIX = "vector_db/" 

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    .main-header {
        background: #1e40af; padding: 25px; border-radius: 12px; text-align: center;
        margin-bottom: 30px; border-bottom: 5px solid #3b82f6;
    }
    .main-header h1 { color: #ffffff !important; font-size: 2.5rem; margin: 0; }
    .main-header p { color: #dbeafe; font-size: 1.1rem; }
    .metric-container {
        background: #f8fafc; border: 2px solid #e2e8f0;
        padding: 20px; border-radius: 12px; text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .metric-label { color: #64748b; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; }
    .metric-value { color: #1e293b; font-size: 2rem; font-weight: 800; }
    .stChatMessage { background-color: #f1f5f9 !important; border: 1px solid #cbd5e1 !important; color: #000000 !important; }
    </style>
    
    <div class="main-header">
        <h1>🛡️ RISK COMMAND CENTER</h1>
        <p>Strategic Multi-Agent Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. AUTH & AWS S3 SYNC ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )
    BUCKET_NAME = st.secrets["S3_BUCKET"]
except Exception as e:
    st.error("🔑 Credentials Missing in Secrets!")
    st.stop()

def sync_from_s3():
    """Download ChromaDB files from S3 to local disk if they exist."""
    if not os.path.exists(LOCAL_DB_PATH):
        os.makedirs(LOCAL_DB_PATH)
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_VECTOR_PREFIX):
            if 'Contents' in result:
                for obj in result['Contents']:
                    rel_path = os.path.relpath(obj['Key'], S3_VECTOR_PREFIX)
                    local_file = os.path.join(LOCAL_DB_PATH, rel_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    s3_client.download_file(BUCKET_NAME, obj['Key'], local_file)
    except Exception as e:
        st.sidebar.warning(f"S3 Sync Note: {e}")

def sync_to_s3():
    """Upload local ChromaDB files back to S3."""
    with st.spinner("Syncing Knowledge Base to Cloud..."):
        for root, _, files in os.walk(LOCAL_DB_PATH):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = os.path.join(S3_VECTOR_PREFIX, os.path.relpath(local_path, LOCAL_DB_PATH))
                s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
        st.sidebar.success("Cloud Sync Complete!")

# --- 3. DATA & VECTOR ENGINE ---
@st.cache_data
def load_data_from_s3(file_key):
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        df = pd.read_csv(obj['Body'])
        df.columns = df.columns.str.strip()
        return df
    except: return pd.DataFrame()

p_df = load_data_from_s3('project_risk_raw_dataset.csv')
m_df = load_data_from_s3('market_trends.csv')
t_df = load_data_from_s3('transaction.csv')

def get_safe_col(df, options):
    for opt in options:
        if opt in df.columns: return opt
    return None

@st.cache_resource
def init_vector_db():
    sync_from_s3()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Initialize Chroma. If local_db_path exists, it loads it.
    v_db = Chroma(
        persist_directory=LOCAL_DB_PATH,
        embedding_function=embeddings,
        collection_name="risk_policies"
    )
    return v_db

vector_db = init_vector_db()

# --- 4. AGENTIC BRAIN ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]

def rag_policy_agent(state: AgentState):
    query = state['messages'][-1].content
    # Retrieve from ChromaDB
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"POLICY CONTEXT:\n{context}\n\nROLE: Policy Compliance Expert. Answer based on context: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Policy_Expert")]}

def manager_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"PROJECT DATA:\n{p_df.describe().to_string()}\n\nROLE: Strategic Risk Manager. Analyze: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Project_Risk_Manager")]}

def market_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"MARKET DATA:\n{m_df.tail(10).to_string()}\n\nROLE: Market Analyst. Analyze: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Market_Analyst")]}

def scoring_agent(state: AgentState):
    query = state['messages'][-1].content
    txn_summary = t_df.groupby('Payment_Status')['Amount_USD'].sum().to_string()
    prompt = f"TRANSACTION DATA:\n{txn_summary}\n\nROLE: Financial Risk Scorer. Analyze: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Risk_Scorer")]}

def router(state: AgentState):
    msg = state['messages'][-1].content.lower()
    # RAG Logic Trigger
    if any(k in msg for k in ["policy", "manual", "rule", "guideline", "compliance"]): return "rag"
    # Existing Data Triggers
    if any(k in msg for k in ["market", "trend", "price", "economy", "inflation", "sentiment"]): return "market"
    if any(k in msg for k in ["transaction", "payment", "overdue", "amount", "score", "default"]): return "scoring"
    return "manager"

builder = StateGraph(AgentState)
builder.add_node("manager", manager_agent)
builder.add_node("market", market_agent)
builder.add_node("scoring", scoring_agent)
builder.add_node("rag", rag_policy_agent)

builder.set_conditional_entry_point(router, {
    "manager": "manager", "market": "market", "scoring": "scoring", "rag": "rag"
})

for node in ["manager", "market", "scoring", "rag"]:
    builder.add_edge(node, END)

agent_brain = builder.compile()

# --- 5. DASHBOARD LAYOUT ---
st.sidebar.header("🛡️ Admin Controls")
if st.sidebar.button("Force Push Vector DB to S3"):
    sync_to_s3()

risk_col = get_safe_col(p_df, ['Risk_Level', 'Risk'])
complexity_col = get_safe_col(p_df, ['Complexity_Score', 'Complexity'])
sentiment_col = get_safe_col(m_df, ['Market_Sentiment', 'Sentiment'])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #ef4444;"><div class="metric-label">Critical Risks</div><div class="metric-value" style="color: #ef4444;">{len(p_df[p_df[risk_col] == "High"]) if risk_col else 0}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #f59e0b;"><div class="metric-label">Knowledge Base Size</div><div class="metric-value" style="color: #f59e0b;">{vector_db._collection.count()} Docs</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #3b82f6;"><div class="metric-label">Avg Complexity</div><div class="metric-value" style="color: #3b82f6;">{p_df[complexity_col].mean():.1f if complexity_col else 0}</div></div>', unsafe_allow_html=True)
with c4:
    last_sent = m_df[sentiment_col].iloc[-1] if sentiment_col and not m_df.empty else 0
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #10b981;"><div class="metric-label">Market Sentiment</div><div class="metric-value" style="color: #10b981;">{last_sent:.2f}</div></div>', unsafe_allow_html=True)

st.write("") 

col_left, col_right = st.columns([3, 2])
with col_left:
    if not p_df.empty:
        fig = px.bar(p_df.head(20), x='Project_ID', y=complexity_col, color=risk_col, title="Project Complexity (Sample)", color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'})
        st.plotly_chart(fig, use_container_width=True)
with col_right:
    if not p_df.empty:
        fig2 = px.pie(p_df, names=risk_col, title="Portfolio Risk", color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'])
        st.plotly_chart(fig2, use_container_width=True)

# --- 6. AGENTIC CHAT ---
st.markdown("<h3 style='color: #1e40af;'>💬 Intelligence Briefing</h3>", unsafe_allow_html=True)
if "history" not in st.session_state: st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Ask about high risks, market trends, or company policies..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.spinner("🤖 Consulting Specialist Agents..."):
        try:
            result = agent_brain.invoke({"messages": [HumanMessage(content=prompt)]})
            ans = result["messages"][-1]
            agent_name = ans.name.replace("_", " ")
            full_msg = f"**{agent_name}**: {ans.content}"
            st.chat_message("assistant").write(full_msg)
            st.session_state.history.append({"role": "assistant", "content": full_msg})
        except Exception as e:
            st.error(f"Error: {e}")
