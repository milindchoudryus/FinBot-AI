# FinBotAI 📈

FinBotAI is a Streamlit-based AI assistant that processes financial news articles from URLs and answers questions using LLMs like Mistral via Ollama. It combines embeddings with retrieval-based QA for finance-focused insights.

## Features
- Load and process news article URLs
- Extract and embed content using sentence-transformers
- Store and retrieve vector data using FAISS
- Ask questions and receive answers powered by Ollama LLMs
- Streamlit UI for interaction

## How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run FinBotAI.py`

## Requirements
- Python 3.8+
- Streamlit
- langchain
- sentence-transformers
- FAISS
- ollama

![FinBot](https://github.com/user-attachments/assets/af60bede-210c-4112-824b-cae4fa3b2eb6)
