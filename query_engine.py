
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

INDEX_DIR = "news_base_openai"

def query_documents(query, placeholder):
    if not os.path.exists(INDEX_DIR):
        placeholder.error("❌ FAISS index not found. Please collect and process data first.")
        return

    embedding_fun = OpenAIEmbeddings()
    vectorstore_base = FAISS.load_local(INDEX_DIR, embedding_fun, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        max_retries=3,
        request_timeout=60
    )

    # Use larger k for better coverage
    retriever = vectorstore_base.as_retriever(search_kwargs={"k": 10})
    retriever_docs = retriever.invoke(query)

    if not retriever_docs:
        placeholder.warning("⚠️ No documents retrieved. Try rephrasing your question.")
        return

    context = "\n\n".join([doc.page_content for doc in retriever_docs])

    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful financial assistant. Use the context below to answer the user's question.

Context:
{context}

Question: {question}

Answer:"""
    )

    formatted_prompt = prompt_template.format(context=context, question=query)

    try:
        response = llm.invoke(formatted_prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        placeholder.error(f"❌ LLM error: {e}")
        return

    if answer:
        st.session_state.last_answer = answer
        placeholder.empty()
        placeholder.header("Answer")
        placeholder.write(answer)
    else:
        st.session_state.last_answer = ""
        placeholder.warning("⚠️ The model did not return a valid answer.")
