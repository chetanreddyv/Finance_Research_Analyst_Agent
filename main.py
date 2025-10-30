from pydantic_ai import Agent, UsageLimits, UsageLimitExceeded
from pydantic import BaseModel, Field
from tavily import TavilyClient
import yfinance as yf
from typing import Optional
from dotenv import load_dotenv
import os


load_dotenv()
TAVILY_API_KEY="xxxxxxxxxxx"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file")


# Initialize Tavily client once
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


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


model_settings = {
    "temperature": 0.3,
    "max_tokens": 2048,
}


agent_system_prompt = (
    "You are a financial analyst assistant. "
    "If the user asks only for latest news, call get_recent_news once and return results; do not call other tools. "
    "If the user asks only for stock analysis, call get_stock_data once and return results. "
    "If the user explicitly asks for both, call each tool once and produce a combined response. "
    "Always return structured, concise outputs and avoid unnecessary follow-up turns."
)


stock_agent = Agent(
    "openai:gpt-4o",
    output_type=StockAnalysisResult,
    model_settings=model_settings,
    system_prompt=agent_system_prompt,
)


@stock_agent.tool_plain
def get_stock_data(symbol: str) -> dict:
    """Fetch comprehensive stock data for analysis."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    output = {
        "symbol": symbol.upper(),
        "current_price": round((info.get('currentPrice') or 0), 2),
        "market_cap": info.get('marketCap'),
        "pe_ratio": info.get('trailingPE'),
        "day_high": info.get('dayHigh'),
        "day_low": info.get('dayLow'),
        "volume": info.get('volume'),
    }
    print("[DEBUG] yf tool output:", output)
    return output


@stock_agent.tool_plain
def get_recent_news(symbol: str, max_results: Optional[int] = 3) -> dict:
    """
    Fetch recent finance news articles related to the stock symbol using Tavily.
    """
    query = f"{symbol} stock news"
    formatted: list[dict] = []

    try:
        # Use the Tavily client's search method
        response = tavily_client.search(
            query=query,
            topic="news",  # Use finance topic for better financial news
            max_results=max_results or 3,
            search_depth="basic",  # Basic is faster and cheaper
            include_answer=False,
            include_raw_content=False,
        )
        
        # Extract results from Tavily response
        for art in (response.get("results") or [])[: max_results or 3]:
            formatted.append({
                "title": art.get("title") or "",
                "summary": art.get("content") or "",  # Tavily uses 'content' for snippets
                "url": art.get("url") or "",
            })
    except Exception as e:
        formatted.append({"title": "Error fetching news", "summary": str(e), "url": ""})

    print("[DEBUG] tavily tool output:", formatted)
    return {"news": formatted}


# For a news-only prompt, tighten limits to 1 tool call
limits = UsageLimits(request_limit=4, tool_calls_limit=2)


try:
    result = stock_agent.run_sync("show me the latest news on tesla", usage_limits=limits)
    print(f"Symbol: {result.output.symbol}")
    print(f"Price: ${result.output.current_price}")
    print(f"Analysis: {result.output.analysis}")
    if result.output.news:
        print("\nRecent News Articles:")
        for article in result.output.news:
            print(f"- {article.title}\n  {article.summary}\n  {article.url}")
except UsageLimitExceeded as e:
    print(f"Usage limit exceeded: {e}")
