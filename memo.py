from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing import Optional, Literal, List
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import yfinance as yf
import asyncio

# Load environment variables
load_dotenv()

# API Keys and Configuration
PINECONE_API_KEY = "xxxxxxxxx"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TAVILY_API_KEY = "xxxxxxxxx"
INDEX_NAME = "sec-rag"
# Validate API keys
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
sec_model = SentenceTransformer(EMBEDDING_MODEL)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ==================== Pydantic Models ====================

class AnalysisIntent(BaseModel):
    """Parsed user intent and execution plan"""
    intent_type: Literal["financials", "news", "valuation", "financials_and_valuation", "news_and_valuation", "financials_and_news","comprehensive"] = Field(description="Type of analysis requested by user")
    company: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    needs_sec_data: bool = Field(description="Whether SEC filing data is needed")
    needs_market_data: bool = Field(description="Whether market/valuation data is needed")
    needs_news: bool = Field(description="Whether news data is needed")
    execution_plan: str = Field(description="Step-by-step plan for data gathering")
        


class GatheredContext(BaseModel):
    """Combined context from all data sources"""
    sec_data: Optional[dict] = Field(default=None, description="SEC filing data")
    market_data: Optional[dict] = Field(default=None, description="Market metrics data")
    news_data: Optional[dict] = Field(default=None, description="News and sentiment data")
    sources_used: List[str] = Field(description="List of data sources actually used")


class FinancialAnalysis(BaseModel):
    """SEC filing analysis results"""
    revenue_trends: str = Field(description="Analysis of revenue trends")
    profitability: str = Field(description="Profitability metrics and analysis")
    cash_flow: str = Field(description="Cash flow from operations analysis")
    balance_sheet: str = Field(description="Balance sheet health assessment")


class KeyMetrics(BaseModel):
    """Key valuation metrics from yfinance"""
    PE: Optional[float] = Field(description="Price to Earnings ratio")
    EV_EBITDA: Optional[float] = Field(description="Enterprise Value to EBITDA")
    PB: Optional[float] = Field(description="Price to Book ratio")
    peer_comparison: Optional[str] = Field(description="Industry and sector context")
    current_price: Optional[float] = Field(description="Current stock price")
    market_cap: Optional[float] = Field(description="Market capitalization")
    revenue_growth_yoy: Optional[str] = Field(description="Year-over-year revenue growth")


class CompanyNews(BaseModel):
    """News and market analysis"""
    summary: str = Field(description="Executive summary of news")
    market_position: str = Field(description="Company's market position")
    recent_developments: str = Field(description="Recent developments and news")
    risks: str = Field(description="Identified risks")
    catalysts: str = Field(description="Potential catalysts")
    source_urls: List[str] = Field(description="Source URLs")


class ExecutiveSummary(BaseModel):
    """Executive summary of investment analysis"""
    company: str = Field(description="Company name and ticker")
    recommendation: Literal["BUY", "HOLD", "SELL"] = Field(description="Investment recommendation")
    target_price: str = Field(description="Target price estimate")
    time_horizon: str = Field(description="Investment time horizon")
    thesis_summary: str = Field(description="Investment thesis summary")
    key_metrics: Optional[KeyMetrics] = Field(description="Key valuation metrics")


class InvestmentMemo(BaseModel):
    """Complete investment memo"""
    executive_summary: ExecutiveSummary
    company_news: Optional[CompanyNews] = None
    financial_analysis: Optional[FinancialAnalysis] = None
    risks: List[str] = Field(description="List of risks")
    catalysts: List[str] = Field(description="List of catalysts")
    analysis_scope: str = Field(description="Scope of analysis performed")


# ==================== Planner Agent ====================

planner_prompt = planner_prompt = """You are an intelligent planning agent for financial analysis.
Your job is to parse user requests and determine EXACTLY which data sources are needed.

## Intent Types and Data Source Mapping

**Single-Focus Intents:**
- "financials": SEC filings, revenue, profitability, cash flow, balance sheet
  → needs_sec_data=true, needs_market_data=false, needs_news=false
  
- "news": Recent news, market sentiment, developments, announcements
  → needs_sec_data=false, needs_market_data=false, needs_news=true
  
- "valuation": Market metrics, P/E ratios, stock price, market cap, comparisons
  → needs_sec_data=false, needs_market_data=true, needs_news=false

**Combination Intents:**
- "financials_and_valuation": SEC data + market metrics for fundamental analysis
  → needs_sec_data=true, needs_market_data=true, needs_news=false
  
- "news_and_valuation": News + market data for sentiment-driven analysis
  → needs_sec_data=false, needs_market_data=true, needs_news=true
  
- "financials_and_news": SEC filings + news for operational + sentiment analysis
  → needs_sec_data=true, needs_market_data=false, needs_news=true

**Complete Analysis:**
- "comprehensive": All data sources for full investment analysis
  → needs_sec_data=true, needs_market_data=true, needs_news=true


## Keyword-to-Intent Mapping Rules

**Trigger "news" data source if query contains:**
- "news", "latest", "recent developments", "sentiment", "market buzz"
- "announcements", "updates", "what's happening", "current events"

**Trigger "market_data" (valuation) if query contains:**
- "valuation", "price", "P/E", "metrics", "market cap", "stock price"
- "trading", "performance", "returns", "market", "sentiment" (+ news)
- "worth", "expensive", "cheap", "multiples"

**Trigger "sec_data" (financials) if query contains:**
- "financials", "revenue", "earnings", "profitability", "cash flow"
- "balance sheet", "SEC", "10-K", "10-Q", "financial performance"
- "fundamentals", "income statement", "operating margin"


## Analysis Examples

**Example 1:**
Query: "What's the latest news and market sentiment on Tesla?"
→ Intent: news_and_valuation
→ needs_sec_data: false
→ needs_market_data: true (sentiment requires current price/market context)
→ needs_news: true
→ Reasoning: "latest news" triggers news, "market sentiment" requires both news + market data

**Example 2:**
Query: "Analyze Apple's financial performance"
→ Intent: financials
→ needs_sec_data: true
→ needs_market_data: false
→ needs_news: false
→ Reasoning: "financial performance" explicitly requests SEC data only

**Example 3:**
Query: "Is Microsoft overvalued?"
→ Intent: financials_and_valuation
→ needs_sec_data: true (need fundamentals to assess valuation)
→ needs_market_data: true (need current multiples)
→ needs_news: false
→ Reasoning: Valuation assessment requires both SEC fundamentals and market metrics

**Example 4:**
Query: "Give me a comprehensive analysis of NVIDIA"
→ Intent: comprehensive
→ needs_sec_data: true
→ needs_market_data: true
→ needs_news: true
→ Reasoning: "comprehensive" explicitly requests all available data

**Example 5:**
Query: "What are analysts saying about Amazon's stock?"
→ Intent: news_and_valuation
→ needs_sec_data: false
→ needs_market_data: true (stock context)
→ needs_news: true (analyst opinions)
→ Reasoning: Analyst opinions are in news, stock context needs market data

**Example 6:**
Query: "Compare Tesla's revenue growth to its competitors"
→ Intent: financials_and_valuation
→ needs_sec_data: true (revenue data)
→ needs_market_data: true (competitive metrics)
→ needs_news: false
→ Reasoning: Comparison requires both SEC financials and market benchmarks


## Your Task

For each user query:
1. **Identify keywords** that signal what data is needed
2. **Map to the appropriate intent type** (single, combination, or comprehensive)
3. **Set data source flags** (needs_sec_data, needs_market_data, needs_news) accurately
4. **Extract company name and ticker** from the query
5. **Create an execution plan** that describes the analysis steps

## Critical Rules

- **Be precise**: Only request data sources that are actually needed
- **Sentiment queries**: Always require BOTH news + market_data (news_and_valuation intent)
- **Valuation queries**: May need SEC data if assessing intrinsic value vs market price
- **News queries**: If asking about stock/market specifically, also include market_data
- **Default to minimum**: When uncertain, prefer narrower intent over comprehensive
- **Combinations over comprehensive**: Use combination intents when possible to minimize unnecessary calls

## Output Format

Return structured data with:
- intent_type: One of the 7 types listed above
- company: Full company name
- ticker: Stock ticker symbol (uppercase)
- needs_sec_data: boolean
- needs_market_data: boolean  
- needs_news: boolean
- execution_plan: Clear step-by-step plan describing what analysis will be performed

Be systematic and precise in your analysis."""


Planner_Agent = Agent(
    "openai:gpt-4o",
    output_type=AnalysisIntent,
    system_prompt=planner_prompt,
)


# ==================== Data Gathering Tools ====================

async def get_sec_data(ticker: str, query: str = "comprehensive SEC analysis") -> dict:
    """Fetch SEC filing data from Pinecone"""
    print(f"\n[get_sec_data] Fetching SEC data for {ticker}")
    
    sections = ["revenue_trends", "profitability", "cash_flow", "balance_sheet"]
    financial_data = {k: "" for k in sections}
    
    combined_query = f"{query} revenue profitability cash flow balance sheet"
    query_embedding = sec_model.encode(combined_query).tolist()
    
    try:
        search_results = index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True,
            filter={"ticker": ticker.upper()}
        )
        
        print(f"[get_sec_data] Found {len(search_results.matches)} matches")
        
        for section in sections:
            for match in search_results.matches:
                meta = match.get("metadata", {})
                text = meta.get("text", "")
                section_name = meta.get("section", "").lower()
                
                if any(keyword in section_name for keyword in section.split("_")) and text:
                    financial_data[section] = text[:512]
                    break
            
            if not financial_data[section]:
                financial_data[section] = f"Limited data available for {section.replace('_', ' ')}"
        
        return financial_data
    except Exception as e:
        print(f"[get_sec_data] ERROR: {e}")
        return {s: "ERROR in SEC query" for s in sections}


async def get_market_data(ticker: str) -> dict:
    """Fetch market and valuation metrics from yfinance"""
    print(f"\n[get_market_data] Fetching market data for {ticker}")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
        
        pe = info.get('trailingPE') or info.get('forwardPE')
        ev = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        ev_ebitda = round(ev / ebitda, 2) if ev and ebitda else None
        
        metrics = {
            "PE": round(pe, 2) if pe else None,
            "EV_EBITDA": ev_ebitda,
            "PB": round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else None,
            "current_price": round(info.get('currentPrice') or info.get('regularMarketPrice', 0), 2),
            "market_cap": info.get('marketCap'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "peer_comparison": f"{info.get('industry', 'N/A')} | {info.get('sector', 'N/A')}"
        }
        
        print(f"[get_market_data] Retrieved metrics: {metrics}")
        return metrics
    except Exception as e:
        print(f"[get_market_data] ERROR: {e}")
        return {"error": str(e)}


async def get_news_data(company: str, ticker: str) -> dict:
    """Fetch recent news from Tavily"""
    print(f"\n[get_news_data] Fetching news for {company} ({ticker})")
    
    try:
        query = f"{company} {ticker} stock news analysis"
        response = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="basic",
            topic="finance",
            include_domains=["bloomberg.com", "reuters.com", "cnbc.com", "marketwatch.com"]
        )
        
        news_items = []
        urls = []
        
        for r in response.get("results", []):
            news_items.append({
                "title": r.get('title', ''),
                "content": r.get('content', '')[:300],
                "url": r.get('url', '')
            })
            urls.append(r.get('url', ''))
        
        result = {
            "news_items": news_items,
            "urls": urls,
            "count": len(news_items)
        }
        
        print(f"[get_news_data] Found {len(news_items)} news articles")
        return result
    except Exception as e:
        print(f"[get_news_data] ERROR: {e}")
        return {"error": str(e), "news_items": [], "urls": []}


# ==================== Context Orchestrator ====================

async def gather_context(intent: AnalysisIntent) -> GatheredContext:
    """Orchestrate data gathering based on planner's intent"""
    print(f"\n[gather_context] Executing plan for '{intent.intent_type}' analysis")
    print(f"[gather_context] Plan: {intent.execution_plan}")
    
    context = GatheredContext(sources_used=[])
    
    # Parallel data gathering for efficiency
    tasks = []
    
    if intent.needs_sec_data:
        print("[gather_context] Queuing SEC data retrieval")
        tasks.append(("sec", get_sec_data(intent.ticker)))
        context.sources_used.append("SEC Filings")
    
    if intent.needs_market_data:
        print("[gather_context] Queuing market data retrieval")
        tasks.append(("market", get_market_data(intent.ticker)))
        context.sources_used.append("Market Data (yfinance)")
    
    if intent.needs_news:
        print("[gather_context] Queuing news retrieval")
        tasks.append(("news", get_news_data(intent.company, intent.ticker)))
        context.sources_used.append("Financial News (Tavily)")
    
    # Execute all data gathering in parallel
    if tasks:
        results = await asyncio.gather(*[task[1] for task in tasks])
        
        for (data_type, _), result in zip(tasks, results):
            if data_type == "sec":
                context.sec_data = result
            elif data_type == "market":
                context.market_data = result
            elif data_type == "news":
                context.news_data = result
    
    print(f"[gather_context] Context gathered from: {', '.join(context.sources_used)}")
    return context
# ==================== Answer Agent ====================
class Answer(BaseModel):
    """Final answer output"""
    analysis: str = Field(description="Comprehensive analysis answer")

answer_prompt = """You are a financial analyst assistant. Your job is to answer user queries using the gathered context.
Be concise and focus on the most relevant information.
"""

Answer_Agent = Agent(
    "openai:gpt-4o-mini",
    output_type=Answer,
    model_settings={"temperature": 0.3, "max_tokens": 1024},
    system_prompt=answer_prompt,
)

# ==================== Analyst Agent ====================

analyst_prompt = """You are a senior financial analyst creating investment memos.
You will receive context from various data sources based on the user's request.

Your job is to synthesize the provided context into a coherent investment analysis.
Focus only on the data sources that were actually gathered - don't make assumptions about missing data.

Provide clear, actionable insights with specific metrics and evidence.
Structure your analysis professionally and cite specific data points.
"""

Analyst_Agent = Agent(
    "openai:gpt-4o",
    output_type=InvestmentMemo,
    model_settings={"temperature": 0.3, "max_tokens": 6000},
    system_prompt=analyst_prompt,
)


# ==================== Main Orchestration ====================

async def generate_investment_memo(user_request: str) -> InvestmentMemo:
    """
    Main function that orchestrates the entire analysis pipeline
    
    Args:
        user_request: Natural language request like 
                     "Give me a comprehensive analysis of Apple"
                     "What's the latest news on Tesla?"
                     "Analyze Microsoft's financial performance"
    """
    print(f"\n{'='*80}")
    print(f"INVESTMENT MEMO GENERATOR")
    print(f"{'='*80}")
    print(f"User Request: {user_request}\n")
    
    # Step 1: Plan the analysis
    print("[STEP 1] Planning analysis...")
    plan_result = await Planner_Agent.run(user_request)
    intent = plan_result.output
    
    print(f"\n[PLANNER OUTPUT]")
    print(f"  Intent Type: {intent.intent_type}")
    print(f"  Company: {intent.company}")
    print(f"  Ticker: {intent.ticker}")
    print(f"  Needs SEC: {intent.needs_sec_data}")
    print(f"  Needs Market: {intent.needs_market_data}")
    print(f"  Needs News: {intent.needs_news}")
    print(f"  Plan: {intent.execution_plan}")
    
    # Step 2: Gather context based on plan
    print(f"\n[STEP 2] Gathering context...")
    context = await gather_context(intent)
    
    # Step 3: Generate analysis
    print(f"\n[STEP 3] Generating investment memo...")
    
    # Build context string for analyst
    context_str = f"""
Analysis Request: {intent.intent_type.upper()} analysis for {intent.company} ({intent.ticker})

Data Sources Used: {', '.join(context.sources_used)}

"""
    
    if context.sec_data:
        context_str += f"\n--- SEC FILING DATA ---\n{context.sec_data}\n"
    
    if context.market_data:
        context_str += f"\n--- MARKET DATA ---\n{context.market_data}\n"
    
    if context.news_data:
        news_summary = "\n".join([
            f"- {item['title']}: {item['content']}" 
            for item in context.news_data.get('news_items', [])
        ])
        context_str += f"\n--- NEWS DATA ---\n{news_summary}\n"
    
    context_str += f"\nCreate a {intent.intent_type} investment memo based on the above context."
    
    # Run analyst agent with gathered context
    memo_result = await Analyst_Agent.run(context_str)
    memo = memo_result.output
    
    # Add metadata about analysis scope
    memo.analysis_scope = f"{intent.intent_type.capitalize()} analysis using: {', '.join(context.sources_used)}"
    
    print(f"\n{'='*80}")
    print(f"MEMO GENERATION COMPLETE")
    print(f"{'='*80}\n")
    
    return memo


# ==================== Helper Functions ====================

def print_memo(memo: InvestmentMemo):
    """Pretty print the investment memo"""
    print("\n" + "="*80)
    print("INVESTMENT MEMO")
    print("="*80)
    
    print(f"\n📊 EXECUTIVE SUMMARY")
    print(f"Company: {memo.executive_summary.company}")
    print(f"Recommendation: {memo.executive_summary.recommendation}")
    print(f"Target Price: {memo.executive_summary.target_price}")
    print(f"Time Horizon: {memo.executive_summary.time_horizon}")
    print(f"\nThesis: {memo.executive_summary.thesis_summary}")
    
    if memo.executive_summary.key_metrics:
        km = memo.executive_summary.key_metrics
        print(f"\n📈 KEY METRICS")
        print(f"Current Price: ${km.current_price}")
        print(f"Market Cap: ${km.market_cap:,}" if km.market_cap else "Market Cap: N/A")
        print(f"P/E Ratio: {km.PE}")
        print(f"EV/EBITDA: {km.EV_EBITDA}")
        print(f"P/B Ratio: {km.PB}")
        print(f"Industry: {km.peer_comparison}")
    
    if memo.financial_analysis:
        print(f"\n💰 FINANCIAL ANALYSIS")
        print(f"Revenue Trends: {memo.financial_analysis.revenue_trends[:200]}...")
        print(f"Profitability: {memo.financial_analysis.profitability[:200]}...")
    
    if memo.company_news:
        print(f"\n📰 NEWS & MARKET POSITION")
        print(f"Summary: {memo.company_news.summary[:200]}...")
        print(f"Recent Developments: {memo.company_news.recent_developments[:200]}...")
    
    print(f"\n⚠️ RISKS")
    for i, risk in enumerate(memo.risks, 1):
        print(f"{i}. {risk}")
    
    print(f"\n🚀 CATALYSTS")
    for i, catalyst in enumerate(memo.catalysts, 1):
        print(f"{i}. {catalyst}")
    
    print(f"\n📋 Analysis Scope: {memo.analysis_scope}")
    print("="*80 + "\n")


# ==================== Example Usage ====================

async def main():
    examples = [
        "Give me a comprehensive investment analysis of Apple Inc",
        "What's the latest news and market sentiment on Tesla?",
        "Analyze Microsoft's financial performance from SEC filings",
        "What are the key valuation metrics for NVIDIA?",
    ]

    print('\nInteractive Investment Memo Generator')
    print('Type a natural language request and press Enter.')
    print("Type 'examples' to list example prompts, or 'quit'/'exit' to stop.\n")

    while True:
        try:
            user_input = input('Request> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting...')
            return

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit'):
            print('Goodbye!')
            return
        if user_input.lower() == 'examples':
            print('\nExamples:')
            for i, ex in enumerate(examples, 1):
                print(f" {i}. {ex}")
            print("\nType the example number to run it, or paste your own request.")
            continue
        # allow selecting an example by number
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            user_input = examples[int(user_input) - 1]

        # Generate and print memo
        print(f"\nRunning analysis for: {user_input}\n")
        try:
            memo = await generate_investment_memo(user_input)
            print_memo(memo)
        except Exception as e:
            print(f"Error generating memo: {e}")
        # small pause to avoid tight loop
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())
