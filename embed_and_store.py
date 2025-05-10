
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def process_and_store_documents(_, placeholder=None):
    input_dir = "top50_stock_data"
    output_dir = "news_base_openai"

    if not os.path.exists(input_dir):
        if placeholder:
            placeholder.error("❌ FAISS index not found. Run data collection first.")
        return

    embedding_fun = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(input_dir, embedding_fun, allow_dangerous_deserialization=True)

    vectorstore.save_local(output_dir)

    if placeholder:
        placeholder.success("✅ Embedding index saved as folder: news_base_openai")
