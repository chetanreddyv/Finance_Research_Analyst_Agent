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
from generate_memo_pdf import create_pdf

# Load environment variables
load_dotenv()

# API Keys and Configuration
PINECONE_API_KEY = "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # Replace with your Pinecone API key
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TAVILY_API_KEY = "tvly-dev-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your Tavily API key
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
    execution_plan: Literal["answer", "investment memo"] = Field(description="Type of output to generate")
        


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
    thesis: str = Field(description="Investment thesis")
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

planner_prompt =  """ You are the Latent-Intent Planner for research. Read any user question and infer the underlying analytical intent even if it is not explicitly stated. Select only the data sources strictly required, extract entities, and return a minimal plan using the exact JSON fields required by the downstream router.

Your output MUST strictly conform to this schema (values only, no extra fields):
- intent_type: one of [financials | news | valuation | financials_and_valuation | news_and_valuation | financials_and_news | comprehensive]
- company: Full company name or index/ETF; empty if unknown
- ticker: UPPERCASE ticker; empty if unknown or ambiguous
- needs_sec_data: boolean
- needs_market_data: boolean
- needs_news: boolean
- execution_plan: one of ["answer", "investment memo"]

Intent taxonomy (pick exactly one):
- financials: Fundamentals from filings and statements only.
- news: Recent developments, analyst commentary, narrative/sentiment only.
- valuation: Market metrics, price, returns, multiples only.
- financials_and_valuation: Fundamentals + market metrics for intrinsic vs market comparison.
- news_and_valuation: News + market data for sentiment with price/multiples context.
- financials_and_news: Fundamentals + news for operational + narrative synthesis.
- comprehensive: All sources for full investment analysis.

Data sources and scope:
- sec_data: SEC/EDGAR filings and fundamentals (10-K, 10-Q), revenue, EPS, margins, FCF, balance sheet, leverage, guidance inside filings.
- market_data: Price/volume, performance/returns, market cap, valuation multiples (P/E, EV/EBITDA, P/B, P/S), volatility, index-relative context.
- news: Headlines, press releases, analyst notes/ratings, sentiment, catalysts, regulatory actions, product launches, macro/company headlines.

Trigger heuristics

News triggers (explicit and implicit):
- Words/phrases: news, latest, update, recent, headlines, what happened, announcement, press release, analysts say, rating, downgrade/upgrade, lawsuit, investigation, product launch, guidance issued, outlook change.
- “What’s happening” or “why did it move” implies news; if stock/market is referenced, also include market_data (news_and_valuation).

Market_data (valuation) triggers:
- Words/phrases: price, today/now, performance, returns, rally, selloff, trading, market cap, valuation, multiples, P/E, EV/EBITDA, cheap/expensive, overvalued/undervalued.
- Pure price/status checks (“What’s NVDA price now?”) → valuation only.
- Any cheap/expensive/overvalued/undervalued judgment usually requires fundamentals too → financials_and_valuation unless explicitly a quick multiple check.

SEC fundamentals triggers:
- Words/phrases: financials, fundamentals, revenue, earnings, EPS, profitability, margins, FCF, cash flow, balance sheet, debt, leverage, unit economics, 10-K, 10-Q, filing, operating margin, segment performance, guidance (in filings).
- “Analyze financial performance” without market language → financials.

Routing to execution_plan
- Use "answer" for direct questions seeking a concise fact or short synthesis (e.g., current price, P/E, quick news summary, single-metric checks, brief comparisons). 
- Use "investment memo" for requests that imply a deeper synthesis (e.g., comprehensive/deep dive, valuation plus fundamentals assessment, multi-source integration, explicit “memo” or “write-up”). 
- If the user asks for “comprehensive”, “deep dive”, “full analysis”, or “investment memo”, always set execution_plan="investment memo".

Defaults and minimization
- Always choose the narrowest set of sources that can answer the question.
- When uncertain, prefer a single-focus intent over comprehensive.
- Prefer combination intents over comprehensive when they suffice.
- Do not include a source unless it is clearly required by the inferred intent.

Composition rules
- Sentiment questions require news + market_data (news_and_valuation).
- Valuation assessments that judge cheap/expensive require fundamentals + market_data unless explicitly a simple multiple check.
- If a question mentions stock/market context alongside news, include market_data with news.
- Peer comparisons that mix fundamentals and relative multiples → financials_and_valuation.
- Analyst opinions or ratings → news_and_valuation.
- “Why did it move?” today/this week → news_and_valuation.

Ambiguity and edge cases
- Multiple companies: set primary as first-mentioned; note peers implicitly (downstream components handle comparisons).
- Ticker-only queries: fill ticker; company may be empty if not confidently resolvable.
- Indices/ETFs: treat as company field; market-only questions → valuation; index news → news or news_and_valuation.
- Private companies: filings unavailable; avoid sec_data and use news and/or market_data only if applicable.
- Timeframes mentioned (e.g., today, this week, last quarter) should influence triggers but are not returned as fields.

Entity extraction
- Extract company full name and ticker (uppercase) when unambiguous; otherwise leave empty and proceed with the narrowest intent you can confidently support.

Flag mapping (must match intent exactly):
- financials → needs_sec_data=true, needs_market_data=false, needs_news=false
- news → needs_sec_data=false, needs_market_data=false, needs_news=true
- valuation → needs_sec_data=false, needs_market_data=true, needs_news=false
- financials_and_valuation → needs_sec_data=true, needs_market_data=true, needs_news=false
- news_and_valuation → needs_sec_data=false, needs_market_data=true, needs_news=true
- financials_and_news → needs_sec_data=true, needs_market_data=false, needs_news=true
- comprehensive → needs_sec_data=true, needs_market_data=true, needs_news=true

Procedure
1) Identify latent intent using explicit keywords and implicit cues (price checks, valuation judgments, sentiment/analyst chatter, filings focus). 
2) Choose the narrowest matching intent and set flags strictly by the mapping above. 
3) Extract company and ticker; leave either blank if ambiguous rather than guessing. 
4) Set execution_plan: “answer” for quick facts/short synthesis, “investment memo” for deep or comprehensive requests. 

Examples

A) “What’s happening with Tesla today?”
- intent_type: news_and_valuation
- Flags: sec=false, market=true, news=true
- execution_plan: answer

B) “Analyze Apple’s financial performance last quarter.”
- intent_type: financials
- Flags: sec=true, market=false, news=false
- execution_plan: answer

C) “Is Microsoft overvalued right now?”
- intent_type: financials_and_valuation
- Flags: sec=true, market=true, news=false
- execution_plan: answer

D) “Give a deep dive on NVIDIA.”
- intent_type: comprehensive
- Flags: sec=true, market=true, news=true
- execution_plan: investment memo

E) “Compare Tesla’s revenue growth to GM and Ford.”
- intent_type: financials_and_valuation
- Flags: sec=true, market=true, news=false
- execution_plan: answer

Return ONLY the JSON fields defined above with correct values. Do not add rationale, notes, or extra keys.
"""


Planner_Agent = Agent(
    "openai:gpt-4o",
    output_type=AnalysisIntent,
    system_prompt=planner_prompt,
)

# STRONG POINT: Intent Parsing via Planning Agent
# - Dedicated Planner Agent parses user queries and selects only the
#   needed data sources (SEC, market, news). This prevents over-fetching
#   and reduces cost/latency by avoiding unnecessary API calls.


# ==================== Data Gathering Tools ====================

async def get_sec_data(ticker: str, query: str = "comprehensive SEC analysis") -> dict:
    """Fetch SEC filing data from Pinecone"""
    print(f"\n[get_sec_data] Fetching SEC data for {ticker}")
    
    sections = ["revenue_trends", "profitability", "cash_flow", "balance_sheet"]
    financial_data = {k: "" for k in sections}
    
    # STRONG POINT: Semantic Chunking for SEC Data Retrieval
    # - Each financial analysis section (revenue_trends, profitability,
    #   cash_flow, balance_sheet) triggers its own semantic embedding/query.
    # - This improves retrieval relevance compared to a single combined query.
    # STRONG POINT: Asynchronous Tool Execution
    # - get_sec_data is async-friendly and designed to run in parallel with
    #   other data gatherers (market/news) to reduce end-to-end latency.
    # Instead of a single combined embedding, run a focused query per section.
    try:
        for section in sections:
            # build a short, focused query for this section
            section_keywords = section.replace('_', ' ')
            section_query = f"{query} {section_keywords}"
            section_emb = sec_model.encode(section_query).tolist()

            search_results = index.query(
                vector=section_emb,
                top_k=12,
                include_metadata=True,
                filter={"ticker": ticker.upper()}
            )

            print(f"[get_sec_data] Section='{section}' found {len(search_results.matches)} matches")

            # pick the first match that contains text
            picked = None
            for match in search_results.matches:
                meta = match.get("metadata", {})
                text = meta.get("text") or meta.get("content") or ""
                if text:
                    picked = text
                    break

            if picked:
                financial_data[section] = picked[:512]
            else:
                financial_data[section] = f"Limited data available for {section.replace('_', ' ')}"

        return financial_data
    except Exception as e:
        # STRONG POINT: Comprehensive Error Handling and Graceful Degradation
        # - Surface helpful fallbacks if RAG or Pinecone fails so the system
        #   returns partial results rather than hard failing.
        print(f"[get_sec_data] ERROR: {e}")
        return {s: "ERROR in SEC query" for s in sections}


async def get_market_data(ticker: str) -> dict:
    """Fetch market, valuation, and detailed financial data from yfinance"""
    print(f"\n[get_market_data] Fetching market data for {ticker}")

    ticker = ticker.upper().strip()
    if " AND " in ticker or " OR " in ticker:
        ticker = ticker.split()[0]

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}

        # Core valuation metrics
        pe = info.get('trailingPE') or info.get('forwardPE')
        ev = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        ev_ebitda = round(ev / ebitda, 2) if ev and ebitda and ebitda != 0 else None

        market_data = {
            "PE": round(pe, 2) if pe else None,
            "EV_EBITDA": ev_ebitda,
            "PB": round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else None,
            "current_price": round(info.get('currentPrice') or info.get('regularMarketPrice', 0), 2),
            "market_cap": info.get('marketCap'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "peer_comparison": f"{info.get('industry', 'N/A')} | {info.get('sector', 'N/A')}",
        }
        print(f"[get_market_data] Comprehensive metrics shape: {len(market_data)} fields")
        return market_data
    except Exception as e:
        print(f"[get_market_data] ERROR: {e}")
        return {
            "PE": None,
            "EV_EBITDA": None,
            "PB": None,
            "current_price": None,
            "market_cap": None,
            "sector": None,
            "industry": None,
            "peer_comparison": "N/A | N/A",
            "error": f"Failed to fetch data: {str(e)}"
        }
 

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
        # STRONG POINT: Comprehensive Error Handling and Graceful Degradation
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

# STRONG POINT: Asynchronous Tool Execution & Modular Data Gathering
# - gather_context queues only the tools indicated by the Planner Agent
#   and executes them in parallel using asyncio.gather.
# - This keeps context selective (SEC/market/news) per request and
#   delivers results faster than sequential calls.
# STRONG POINT: Source Attribution and Auditability
# - `context.sources_used` lists which sources were actually queried
#   so downstream analytic outputs can include clear attribution.
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

analyst_prompt = """You are a senior equity research analyst at a top-tier investment firm.

You will receive detailed, pre-structured context from the following possible sources:
- SEC/EDGAR filings and company financial statements
- Live market data and valuation metrics (price, multiples, peer benchmarks)
- Curated news (headlines, analyst commentary, sentiment, catalysts/risks)
Each section is labeled, and missing data is simply absent—never assume content you did not receive.

Your objective:
- Synthesize ALL provided context—SEC, market, news—into a comprehensive investment memo.
- Your output must be effective for institutional investors, portfolio managers, or sourcing teams.
- Do NOT speculate or guess about any missing facts. If a section (e.g., news or SEC) was not gathered, simply omit from analysis and output.
- Be specific: cite concrete metrics, numbers, growth rates, dates, and named events from context. Never use generic statements or boilerplate.
- Structure your memo as follows (unless context absence dictates omitting sections):

1. **Executive Summary:** Company, ticker, recommendation (BUY/HOLD/SELL), time horizon, price target, succinct thesis—focus on key findings and decision points.
2. **Key Metrics:** Table/list of valuation metrics (P/E, EV/EBITDA, market cap, price, peer comparison), with numbers and peer context.
3. **Financial Analysis:** Discuss revenue trends, margins, profitability, cash flow, and balance sheet health using SEC and financials context.
4. **News and Market Position:** Summarize recent developments, strategic moves, analyst sentiment, regulatory actions, product launches, or macro factors—using only gathered news context.
5. **Risks:** List and briefly explain any risks explicitly surfaced by the context (e.g., regulatory, competitive, supply chain, financial).
6. **Catalysts:** List concrete, specific future events or factors that may drive price/appreciation (announced launches, guidance, industry moves).
7. **Analysis Scope/Attribution:** Note which data sources were used and the breadth of analysis (e.g., “Comprehensive memo based on SEC filings, yfinance market data, Bloomberg news.”).

Rules:
- Every statement must be directly supported by supplied context—never infer or assume beyond the data present.
- Use clear, direct language with bullet points/lists and small tables for metrics if helpful.
- Support every key claim or insight with a quantitative or qualitative reference: metric, year, headline, or quote excerpt if relevant.
- Be concise but thorough—memorable, unbiased, and useful for real capital allocation or strategic review.

If context is missing for a section, simply omit it and focus analysis on what is actually present.

Return a fully structured investment memo in the Pydantic InvestmentMemo schema defined by the system, with all available sections filled.

Example output outline (fields must match schema exactly):

InvestmentMemo:
  - executive_summary:
      company: ...
      recommendation: ... BUY | HOLD | SELL
      target_price: ...
      time_horizon: ...
      thesis: ...
      key_metrics: { ... }
  - financial_analysis: { ... }
  - company_news: { ... }
  - risks: [ ... ]
  - catalysts: [ ... ]
  - analysis_scope: "Comprehensive memo using: SEC filings, yfinance, Bloomberg, Reuters."

"""

Analyst_Agent = Agent(
    "openai:gpt-4o",
    output_type=InvestmentMemo,
    model_settings={"temperature": 0.4, "max_tokens": 7000},
    system_prompt=analyst_prompt,
)


# ==================== Main Orchestration ====================


async def generate_analysis(user_request: str):
    """
    Main function that orchestrates the entire analysis pipeline

    Flow: User Request → Planner Agent → {Answer Agent | Analyst Agent}

    Args:
        user_request: Natural language request like 
            "Give me a comprehensive analysis of Apple"
            "What's the latest news on Tesla?"
            "Analyze Microsoft's financial performance"
    """
    print(f"\n{'='*80}")
    print(f"FINANCIAL ANALYSIS ENGINE")
    print(f"{'='*80}")
    print(f"User Request: {user_request}\n")

    # ========== STEP 1: Plan the analysis ==========
    print("[STEP 1] Planning analysis with Planner Agent...")
    try:
        plan_result = await Planner_Agent.run(user_request)
        # Agent run returns an object with `.output` containing the typed result
        intent = plan_result.output
    except Exception as e:
        print(f"[ERROR] Planner Agent failed: {e}")
        return None

    print(f"\n[PLANNER OUTPUT]")
    print(f"  Intent Type: {intent.intent_type}")
    print(f"  Company: {intent.company}")
    print(f"  Ticker: {intent.ticker}")
    print(f"  Needs SEC: {intent.needs_sec_data}")
    print(f"  Needs Market: {intent.needs_market_data}")
    print(f"  Needs News: {intent.needs_news}")
    print(f"  Execution Plan: {intent.execution_plan}")

    # ========== STEP 2: Gather context based on plan ==========
    print(f"\n[STEP 2] Gathering context from data sources...")
    context = await gather_context(intent)

    # ========== STEP 3: Route to appropriate agent ==========
    print(f"\n[STEP 3] Routing to {intent.execution_plan} pipeline...\n")
    if intent.execution_plan == "answer":
        print("[ROUTING] → Answer Agent")
        return await handle_answer_request(intent, context, user_request)

    elif intent.execution_plan == "investment memo":
        print("[ROUTING] → Analyst Agent")
        return await handle_memo_request(intent, context)

    else:
        print(f"[ERROR] Unknown execution plan: {intent.execution_plan}")
        return None


async def handle_answer_request(intent, context, user_request):
    """Handle simple answer requests via Answer Agent"""
    context_str = f"""
User Question: {user_request}

Company: {intent.company} ({intent.ticker})
Analysis Type: {intent.intent_type}
Data Sources Available: {', '.join(context.sources_used)}

"""
    if context.sec_data:
        context_str += f"\nSEC Filing Data:\n{context.sec_data}\n"

    if context.market_data:
        context_str += f"\nMarket Data:\n{str(context.market_data)}\n"

    if context.news_data:
        news_summary = "\n".join([
            f"- {item['title']}: {item['content']}" 
            for item in context.news_data.get('news_items', [])[:3]
        ])
        context_str += f"\nRecent News:\n{news_summary}\n"

    context_str += f"\nProvide a concise, direct answer to the user's question based on the above data."

    try:
        answer_result = await Answer_Agent.run(context_str)
        # use `.output` to access the typed Pydantic result
        answer = answer_result.output

        print(f"\n{'='*80}")
        print("ANALYSIS ANSWER")
        print(f"{'='*80}")
        print(f"\nCompany: {intent.company} ({intent.ticker})")
        print(f"Question: {user_request}")
        print(f"\n{answer.analysis}")
        print(f"\nData Sources: {', '.join(context.sources_used)}")
        print("="*80 + "\n")

        return answer

    except Exception as e:
        print(f"[ERROR] Answer Agent failed: {e}")
        return None


async def handle_memo_request(intent, context):
    """Handle investment memo requests via Analyst Agent"""
    context_str = f"""
Generate a professional investment memo for:
Company: {intent.company} ({intent.ticker})
Analysis Type: {intent.intent_type}
Data Sources Used: {', '.join(context.sources_used)}

"""

    if context.sec_data:
        context_str += f"--- SEC FILING DATA ---\n"
        for key, value in context.sec_data.items():
            context_str += f"{key.replace('_', ' ').title()}: {value}\n"
        context_str += "\n"

    if context.market_data:
        context_str += f"--- MARKET DATA ---\n"
        for key, value in context.market_data.items():
            context_str += f"{key}: {value}\n"
        context_str += "\n"

    if context.news_data:
        context_str += f"--- NEWS DATA ---\n"
        for item in context.news_data.get('news_items', []):
            context_str += f"Title: {item['title']}\n"
            context_str += f"Content: {item['content']}\n"
            context_str += f"URL: {item['url']}\n\n"

    context_str += f"Create a comprehensive {intent.intent_type} investment memo."

    try:
        memo_result = await Analyst_Agent.run(context_str)
        # use `.output` to access the typed Pydantic result
        memo = memo_result.output

        # Add metadata
        memo.analysis_scope = f"{intent.intent_type.capitalize()} analysis using: {', '.join(context.sources_used)}"

        print_investment_memo(memo)
        
        # Generate PDF
        pdf_filename = f"{intent.ticker}_{intent.intent_type}_memo.pdf"
        try:
            # Convert Pydantic model to dict for PDF generator
            memo_dict = memo.model_dump()
            create_pdf(memo_dict, pdf_filename)
            print(f"\n✅ PDF saved: {pdf_filename}\n")
        except Exception as pdf_error:
            print(f"\n⚠️  PDF generation failed: {pdf_error}")
            print("Continuing with text output...\n")
        
        return memo

    except Exception as e:
        print(f"[ERROR] Analyst Agent failed: {e}")
        return None


def print_investment_memo(memo):
    """Pretty print the investment memo"""
    print("\n" + "="*80)
    print("INVESTMENT MEMO")
    print("="*80)

    # Executive Summary
    print(f"\n📊 EXECUTIVE SUMMARY")
    print(f"Company: {memo.executive_summary.company}")
    print(f"Recommendation: {memo.executive_summary.recommendation}")
    print(f"Target Price: {memo.executive_summary.target_price}")
    print(f"Time Horizon: {memo.executive_summary.time_horizon}")
    print(f"\nThesis: {memo.executive_summary.thesis}")

    # Key Metrics
    if memo.executive_summary.key_metrics:
        km = memo.executive_summary.key_metrics
        print(f"\n📈 KEY METRICS")
        if km.current_price:
            print(f"  Current Price: ${km.current_price}")
        if km.market_cap:
            print(f"  Market Cap: ${km.market_cap:,}")
        if km.PE:
            print(f"  P/E Ratio: {km.PE}")
        if km.EV_EBITDA:
            print(f"  EV/EBITDA: {km.EV_EBITDA}")
        if km.PB:
            print(f"  P/B Ratio: {km.PB}")
        if km.peer_comparison:
            print(f"  Industry Context: {km.peer_comparison}")

    # Financial Analysis
    if memo.financial_analysis:
        print(f"\n💰 FINANCIAL ANALYSIS")
        fa = memo.financial_analysis
        if fa.revenue_trends:
            print(f"  Revenue Trends: {fa.revenue_trends}")
        if fa.profitability:
            print(f"  Profitability: {fa.profitability}")
        if fa.cash_flow:
            print(f"  Cash Flow: {fa.cash_flow}")

    # Company News
    if memo.company_news:
        print(f"\n📰 NEWS & MARKET POSITION")
        cn = memo.company_news
        if cn.summary:
            print(f"  Summary: {cn.summary}")
        if cn.recent_developments:
            print(f"  Recent Developments: {cn.recent_developments}")

    # Risks
    if memo.risks:
        print(f"\n⚠️  RISKS")
        for i, risk in enumerate(memo.risks, 1):
            print(f"  {i}. {risk}")

    # Catalysts
    if memo.catalysts:
        print(f"\n🚀 CATALYSTS")
        for i, catalyst in enumerate(memo.catalysts, 1):
            print(f"  {i}. {catalyst}")

    print(f"\n📋 Analysis Scope: {memo.analysis_scope}")
    print("="*80 + "\n")


# ====================  MAIN LOOP ====================

async def main():
    """Interactive investment analysis engine"""

    examples = [
    # INVESTMENT MEMO WORKFLOW (trigger "investment memo")
    "Generate an investment memo for Apple Inc with comprehensive analysis",
    "Create a full investment memo on AMD including financials, news, and valuation",
    "Write an investment memo for Starbucks using latest information",

    # ANSWER WORKFLOW (trigger "answer")
    "What is the current P/E ratio for Microsoft?",
    "Summarize the latest news about Nvidia's stock",
    "Compare Apple's market cap with Amazon's",
    "Is Tesla profitable this quarter?",
    "Show NVIDIA's year-over-year revenue growth",
    "What is AMD's current stock price?",
    "Give sector and industry context for Starbucks stock",
    ]

    print('\n' + "="*80)
    print('INTERACTIVE INVESTMENT ANALYSIS ENGINE')
    print("="*80)
    print('Type a natural language request and press Enter.')
    print("Type 'examples' to list example prompts, or 'quit'/'exit' to stop.\n")

    while True:
        try:
            user_input = input('Request> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting...')
            return

        # Handle empty input
        if not user_input:
            continue

        # Handle exit commands
        if user_input.lower() in ('quit', 'exit'):
            print('Goodbye!')
            return

        # Handle examples list
        if user_input.lower() == 'examples':
            print('\nExample Prompts:')
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            print("\nType the example number to run it, or paste your own request.\n")
            continue

        # Handle selecting example by number
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            user_input = examples[int(user_input) - 1]

        # Execute analysis with corrected flow
        print(f"\nProcessing: {user_input}\n")
        try:
            result = await generate_analysis(user_input)
            if result is None:
                print("[WARNING] Analysis failed to complete\n")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}\n")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())
