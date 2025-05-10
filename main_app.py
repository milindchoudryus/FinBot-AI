
from dotenv import load_dotenv
import os
import streamlit as st
import subprocess
from embed_and_store import process_and_store_documents
from query_engine import query_documents

# Load OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="FinBot AI", page_icon="📈")

# Styling and layout
st.markdown(
    """
    <style>
    body {
        background-color: #dceef2;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 6rem;
    }
    header {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 10px 0;
        text-align: center;
        font-size: 0.85rem;
        color: #4b4b4b;
        background-color: #c9e4f7;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("https://cdn-icons-png.flaticon.com/512/2891/2891605.png", width=100)
st.title("FinBot AI 📈")
st.caption("Smart Financial Article Analysis with LLMs")

main_placeholder = st.empty()

# Combined action: collect data + process embeddings
if st.button("📥 Collect Data"):
    with st.spinner("Collecting stock data and embedding with OpenAI..."):
        subprocess.run(["python3", "top50_stock_embedder.py"])
        process_and_store_documents(None, main_placeholder)
    st.success("✅ Stock data collected and FAISS index saved as 'news_base_openai/'")

# Query input
query = st.text_input("Enter your financial question:")
ask_button = st.button("Ask")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if ask_button and query:
    with st.spinner("🔍 Performing Semantic Search & Generating Answer..."):
        query_documents(query, main_placeholder)

# Footer
st.markdown(
    """
    <div class="footer">
        ⚙️ Built using OpenAI, LangChain, FAISS and Streamlit |
        🚀 Version: 1.0 | 👨‍💻 Made with ❤️ by FinBot AI
    </div>
    """,
    unsafe_allow_html=True
)
