from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.google import GoogleModelSettings
import yfinance as yf
from typing import Optional
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Verify the API key is loaded (optional)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


class StockAnalysisResult(BaseModel):
    symbol: str
    current_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None
    analysis: str = Field(description="Brief analysis of the stock")


# Optional: Configure model settings
model_settings = GoogleModelSettings(
    temperature=0.2,
    max_tokens=1024,  # If your installed version expects max_output_tokens, swap the name accordingly
    google_thinking_config={'thinking_budget': 0}  # Disable thinking to save costs
)

stock_agent = Agent(
    "google-gla:gemini-2.5-flash",
    output_type=StockAnalysisResult,
    model_settings=model_settings,
    system_prompt=(
        "You are a financial analyst assistant. "
        "Use the get_stock_data tool to fetch comprehensive stock information and provide insights."
    ),
)

@stock_agent.tool_plain
def get_stock_data(symbol: str) -> dict:
    """Fetch comprehensive stock data for analysis."""
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "current_price": round(info.get('currentPrice', 0) or 0, 2),
        "market_cap": info.get('marketCap'),
        "pe_ratio": info.get('trailingPE'),
        "day_high": info.get('dayHigh'),
        "day_low": info.get('dayLow'),
        "volume": info.get('volume'),
    }

# Run with various prompts
result = stock_agent.run_sync("Analyze Tesla stock for me")

# Access the structured output via `result.output`
print(f"Symbol: {result.output.symbol}")
print(f"Price: ${result.output.current_price}")
print(f"Analysis: {result.output.analysis}")
