from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.google import GoogleModelSettings
import yfinance as yf
from typing import Optional
from dotenv import load_dotenv
import os
import requests

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Should be set in .env
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

class NewsArticle(BaseModel):
    title: str
    summary: str
    url: str

class StockAnalysisResult(BaseModel):
    symbol: str
    current_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None
    analysis: str = Field(description="Brief analysis of the stock")
    news: Optional[list[NewsArticle]] = Field(default=None, description="Recent related finance news articles.")

model_settings = GoogleModelSettings(
    temperature=0.2,
    max_tokens=2048,
    google_thinking_config={"thinking_budget": 0}
)

# The system prompt now describes both tools and how/when to use them:
agent_system_prompt = (
    "You are a financial analyst assistant. "
    "Use get_stock_data for comprehensive stock information and insights. "
    "Use get_recent_news when a prompt requests news or recent articles. "
    "When a prompt mentions both stock analysis and news, use both tools and return a combined result. "
    "Always return structured, concise outputs."
)

stock_agent = Agent(
    "google-gla:gemini-2.5-flash",
    output_type=StockAnalysisResult,
    model_settings=model_settings,
    system_prompt=agent_system_prompt,
)

@stock_agent.tool_plain
def get_stock_data(symbol: str) -> dict:
    """Fetch comprehensive stock data for analysis."""
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "symbol": symbol,
        "current_price": round(info.get('currentPrice', 0) or 0, 2),
        "market_cap": info.get('marketCap'),
        "pe_ratio": info.get('trailingPE'),
        "day_high": info.get('dayHigh'),
        "day_low": info.get('dayLow'),
        "volume": info.get('volume'),
    }

@stock_agent.tool_plain
def get_recent_news(symbol: str, max_results: Optional[int] = 3) -> dict:
    """
    Fetch recent finance news articles related to the stock symbol.
    Uses Tavily dev plan API (https://api.tavily.com/dev/search), default max_results=3 unless specified.
    """
    query = f"{symbol} stock news"
    try:
        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}"
        }
        payload = {
            "query": query,
            "max_results": max_results
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        articles = resp.json().get("results", [])
    except Exception as e:
        articles = [{"title": "Error fetching news", "description": str(e), "url": ""}]
    
    formatted = []
    for art in articles:
        formatted.append({
            "title": art.get("title", ""),
            "summary": art.get("description", ""),
            "url": art.get("url", "")
        })
    return {
        "news": formatted
    }


# Example prompt: "Analyze Tesla stock and show me the latest news"
result = stock_agent.run_sync("show me the latest news on tesla")

print(f"Symbol: {result.output.symbol}")
print(f"Price: ${result.output.current_price}")
print(f"Analysis: {result.output.analysis}")
if result.output.news:
    print("\nRecent News Articles:")
    for article in result.output.news:
        print(f"- {article.title}\n  {article.summary}\n  {article.url}")

print("Loaded TAVILY_API_KEY:", TAVILY_API_KEY)
