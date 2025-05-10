
import os
import yfinance as yf
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load OpenAI key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Top 50 tickers
top_50_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "UNH", "PG", "MA", "HD", "DIS", "BAC", "PFE", "VZ", "KO", "INTC",
    "PEP", "T", "CSCO", "WMT", "ADBE", "NFLX", "CRM", "ABT", "ORCL", "XOM",
    "COST", "MCD", "CVX", "LLY", "MRK", "WFC", "TXN", "BA", "AMD", "NKE",
    "IBM", "UPS", "HON", "GE", "GS", "BLK", "CAT", "SBUX", "AMGN", "DE"
]

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=7)

documents = []

for symbol in top_50_tickers:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        info = ticker.info
        price_summary = hist.to_string()
        description = info.get("longBusinessSummary", "")
        sector = info.get("sector", "")
        doc_text = f"Symbol: {symbol}\nSector: {sector}\n\nDescription:\n{description}\n\nWeekly Price Data:\n{price_summary}"
        documents.append(Document(page_content=doc_text, metadata={"symbol": symbol}))
    except Exception as e:
        print(f"Failed to retrieve data for {symbol}: {e}")

print(f"✅ Retrieved and constructed {len(documents)} documents.")

embedding_fun = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_fun)
vectorstore.save_local("top50_stock_data")

print("✅ FAISS vectorstore saved to 'top50_stock_data/'")
