import os
import streamlit as st
import pickle
import time
import ollama
import pickle
import numpy as np
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from chromadb.api.types import EmbeddingFunction
from langchain.llms import Ollama

# class ChromaEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#
#     def __call__(self, texts):
#         return self.embedding_model.embed_documents(texts)

# CHROMA_DB_PATH = "chroma_db"
MAX_CHUNK_SIZE = 512
TOP_K = 5
D = 768
NPROBE = 5

st.title("FinBot AI 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "news_base.pkl"
#
main_placeholder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    url_texts = [doc.page_content for doc in data]

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=50
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_text("\n".join(url_texts))

    documents = [Document(page_content=chunk, metadata={"source": urls[i % len(urls)]}) for i, chunk in enumerate(docs)]

    # print (documents)

    # embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    # vectorstore_base = [embed_model.encode(doc) for doc in docs]
    # main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)

    embedding_fun = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embedding_fun = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    # embedding_function = ChromaEmbeddingFunction()
    vectorstore_base = FAISS.from_documents(documents, embedding_fun)


    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_base, f)
    main_placeholder.text("Embeddings Stored Successfully! ✅✅✅")


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore_base = pickle.load(f)

        # Initialize LLaMA 2 via Ollama
        # llm = Ollama(model="llama2")
        llm = Ollama(model="mistral")

        # Set up RetrievalQAWithSourcesChain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_base.as_retriever())

        # Get the answer
        result = chain({"question": query}, return_only_outputs=True)

        # Display Answer
        st.header("Answer")
        st.write(result["answer"])

        # Display Sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)


