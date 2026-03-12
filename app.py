import os
import boto3
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from typing import Annotated, TypedDict, List

# LangGraph + RAG
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Risk Intel Pro",
    layout="wide",
    page_icon="🛡️"
)

LOCAL_DB_PATH = "/tmp/vector_db"
S3_VECTOR_PREFIX = "vector_db/"

# -------------------------------
# 2. UI STYLING
# -------------------------------

st.markdown("""
<style>

.stApp{
background-color:#ffffff;
color:#1e293b;
}

.main-header{
background:#1e40af;
padding:25px;
border-radius:12px;
text-align:center;
margin-bottom:30px;
border-bottom:5px solid #3b82f6;
}

.main-header h1{
color:white !important;
font-size:2.5rem;
}

.metric-container{
background:#f8fafc;
border:2px solid #e2e8f0;
padding:20px;
border-radius:12px;
text-align:center;
box-shadow:2px 2px 10px rgba(0,0,0,0.05);
}

.metric-label{
color:#64748b;
font-weight:600;
font-size:0.9rem;
text-transform:uppercase;
}

.metric-value{
color:#1e293b;
font-size:2rem;
font-weight:800;
}

</style>

<div class="main-header">
<h1>🛡️ RISK COMMAND CENTER</h1>
<p>Strategic Multi-Agent Intelligence Dashboard</p>
</div>

""", unsafe_allow_html=True)

# -------------------------------
# 3. AUTH + AWS CONFIG
# -------------------------------

try:

    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    BUCKET_NAME = st.secrets["S3_BUCKET"]

except Exception:
    st.error("🔑 Missing credentials in Streamlit secrets")
    st.stop()

# -------------------------------
# 4. S3 VECTOR DB SYNC
# -------------------------------

def sync_from_s3():

    if not os.path.exists(LOCAL_DB_PATH):
        os.makedirs(LOCAL_DB_PATH)

    try:

        paginator = s3_client.get_paginator("list_objects_v2")

        for result in paginator.paginate(
            Bucket=BUCKET_NAME,
            Prefix=S3_VECTOR_PREFIX
        ):

            if "Contents" in result:

                for obj in result["Contents"]:

                    rel_path = os.path.relpath(obj["Key"], S3_VECTOR_PREFIX)

                    local_file = os.path.join(LOCAL_DB_PATH, rel_path)

                    os.makedirs(os.path.dirname(local_file), exist_ok=True)

                    s3_client.download_file(
                        BUCKET_NAME,
                        obj["Key"],
                        local_file
                    )

    except Exception as e:
        st.sidebar.warning(f"S3 Sync issue: {e}")

# -------------------------------
# 5. LOAD DATA
# -------------------------------

@st.cache_data
def load_data_from_s3(file_key):

    try:

        obj = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=file_key
        )

        df = pd.read_csv(obj["Body"])

        df.columns = df.columns.str.strip()

        return df

    except:

        return pd.DataFrame()

p_df = load_data_from_s3("project_risk_raw_dataset.csv")
m_df = load_data_from_s3("market_trends.csv")
t_df = load_data_from_s3("transaction.csv")

# -------------------------------
# 6. VECTOR DATABASE
# -------------------------------

@st.cache_resource
def init_vector_db():

    sync_from_s3()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview"
    )

    vectordb = Chroma(
        persist_directory=LOCAL_DB_PATH,
        embedding_function=embeddings,
        collection_name="risk_policies"
    )

    return vectordb

vector_db = init_vector_db()

# -------------------------------
# 7. LLM
# -------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1
)

# -------------------------------
# 8. AGENT STATE
# -------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "history"]

# -------------------------------
# 9. AGENTS
# -------------------------------

def rag_policy_agent(state: AgentState):

    query = state["messages"][-1].content

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
POLICY CONTEXT
{context}

ROLE: Policy Compliance Expert

Answer the user question using the policy context.

QUESTION:
{query}
"""

    res = llm.invoke(prompt)

    return {"messages": [AIMessage(content=res.content, name="Policy_Expert")]}


def manager_agent(state: AgentState):

    query = state["messages"][-1].content

    data_context = f"""
STATS:
{p_df.describe().to_string()}

SAMPLE:
{p_df.head(10).to_string()}
"""

    prompt = f"""
PROJECT DATA
{data_context}

ROLE: Strategic Risk Manager

Analyze the question:

{query}
"""

    res = llm.invoke(prompt)

    return {"messages": [AIMessage(content=res.content, name="Project_Risk_Manager")]}


def market_agent(state: AgentState):

    query = state["messages"][-1].content

    prompt = f"""
MARKET DATA
{m_df.tail(10).to_string()}

ROLE: Market Analyst

Analyze:

{query}
"""

    res = llm.invoke(prompt)

    return {"messages": [AIMessage(content=res.content, name="Market_Analyst")]}


def scoring_agent(state: AgentState):

    query = state["messages"][-1].content

    txn_summary = (
        t_df.groupby("Payment_Status")["Amount_USD"].sum().to_string()
        if not t_df.empty else "No transaction data"
    )

    prompt = f"""
TRANSACTION SUMMARY
{txn_summary}

ROLE: Financial Risk Scorer

Analyze:

{query}
"""

    res = llm.invoke(prompt)

    return {"messages": [AIMessage(content=res.content, name="Risk_Scorer")]}

# -------------------------------
# 10. ROUTER
# -------------------------------

def router(state: AgentState):

    msg = state["messages"][-1].content.lower()

    if any(k in msg for k in ["policy","rule","manual","guideline"]):
        return "rag"

    if any(k in msg for k in ["market","trend","inflation","economy"]):
        return "market"

    if any(k in msg for k in ["payment","transaction","amount","default"]):
        return "scoring"

    return "manager"

# -------------------------------
# 11. BUILD GRAPH
# -------------------------------

builder = StateGraph(AgentState)

builder.add_node("manager", manager_agent)
builder.add_node("market", market_agent)
builder.add_node("scoring", scoring_agent)
builder.add_node("rag", rag_policy_agent)

builder.set_conditional_entry_point(
    router,
    {
        "manager":"manager",
        "market":"market",
        "scoring":"scoring",
        "rag":"rag"
    }
)

for node in ["manager","market","scoring","rag"]:
    builder.add_edge(node, END)

agent_brain = builder.compile()

# -------------------------------
# 12. CLEAN RESPONSE FUNCTION
# -------------------------------

def clean_response(content):

    if isinstance(content, dict):
        return content.get("text", str(content))

    if isinstance(content, list):

        text_blocks = []

        for block in content:

            if isinstance(block, dict) and "text" in block:
                text_blocks.append(block["text"])
            else:
                text_blocks.append(str(block))

        return "\n".join(text_blocks)

    return str(content)

# -------------------------------
# 13. CHAT UI
# -------------------------------

st.markdown("### 💬 Intelligence Briefing")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about project risks, transactions, or market trends..."):

    st.session_state.history.append(
        {"role":"user","content":prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Consulting Specialist Agents..."):

        try:

            result = agent_brain.invoke(
                {"messages":[HumanMessage(content=prompt)]}
            )

            ans = result["messages"][-1]

            agent_name = (
                ans.name.replace("_"," ")
                if hasattr(ans,"name")
                else "Assistant"
            )

            clean_text = clean_response(ans.content)

            final_output = f"""
### 🧠 {agent_name}

{clean_text}
"""

            with st.chat_message("assistant"):
                st.markdown(final_output)

            st.session_state.history.append(
                {
                    "role":"assistant",
                    "content":final_output
                }
            )

        except Exception as e:

            st.error(f"Agent error: {e}")
