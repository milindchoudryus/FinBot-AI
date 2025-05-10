# 📈 FinBot AI

FinBot AI is an interactive Streamlit-based app that lets you:
- ✅ Collect stock and financial summaries for top 50 companies from Yahoo Finance
- ✅ Embed this data using OpenAI embeddings
- ✅ Query the data using GPT-3.5 with semantic search powered by FAISS

---

## 🚀 Features

- One-click stock data collection via `📥 Collect Data` button
- Uses OpenAI Embeddings + FAISS for efficient semantic search
- Clean, modern interface with custom styling
- Displays natural language answers to your financial queries

---

## 🛠 Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup

1. Create a `.env` file in the root directory:
    ```
    OPENAI_API_KEY=your-openai-api-key
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run main_app.py
    ```

---

## 📁 Project Structure

```
.
├── main_app.py               # Streamlit frontend
├── top50_stock_embedder.py  # Collects and embeds stock data
├── embed_and_store.py       # Converts stock data to FAISS index
├── query_engine.py          # Query logic for semantic search
├── requirements.txt
├── .env                     # Your OpenAI API key (not included here)
└── top50_stock_data/        # Auto-created FAISS store (raw)
└── news_base_openai/        # Final FAISS index for querying
```

---

## 🤖 Model Info

- Embeddings: `OpenAIEmbeddings` (MiniLM-style from OpenAI API)
- LLM: `gpt-3.5-turbo` via `ChatOpenAI`

---

## 🙋 FAQ

**Q: Where does data come from?**  
A: Yahoo Finance via the `yfinance` Python package.

**Q: How do I re-embed data?**  
A: Just click **📥 Collect Data** again!

---

## 👨‍💻 Built With

- [Streamlit](https://streamlit.io)
- [OpenAI API](https://platform.openai.com)
- [LangChain](https://www.langchain.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [yFinance](https://github.com/ranaroussi/yfinance)

---

## 📬 Contact

Made with ❤️ by FinBot AI. For issues or suggestions, reach out via email or GitHub.