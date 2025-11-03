from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from datetime import date
import os

# ------ Theme & Styles ------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontName="Helvetica-Bold", fontSize=18, spaceAfter=6, textColor=colors.HexColor("#222")))
styles.add(ParagraphStyle(name="Subtitle", fontName="Helvetica", fontSize=13, textColor=colors.HexColor("#555"), spaceAfter=10))
styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=13, spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#222")))
styles.add(ParagraphStyle(name="SectionSmall", fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#222")))
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=10.5, leading=13, textColor=colors.HexColor("#222"), spaceAfter=2))
styles.add(ParagraphStyle(name="CustomBullet", fontName="Helvetica", fontSize=10, leftIndent=12, bulletIndent=0, spaceAfter=2))
styles.add(ParagraphStyle(name="Muted", fontName="Helvetica-Oblique", fontSize=9, textColor=colors.HexColor("#777")))

def _human_currency(v):
    try:
        n = float(v)
    except Exception:
        return str(v)
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.2f}M"
    if n >= 1e3: return f"${n/1e3:.2f}K"
    return f"${n:.2f}"

# ------ Section Builders ------
def add_cover(story, memo):
    ex = memo.get("executive_summary", {}) or {}
    company = ex.get("company") or memo.get("company") or "Company"
    rec = ex.get("recommendation") or ""
    price = ex.get("target_price") or ""
    horizon = ex.get("time_horizon") or ""
    thesis = ex.get("thesis") or memo.get("company_news", {}).get("summary") or ""
    sub = " • ".join([x for x in [rec, price, horizon] if x])
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(f"Investment Memo – {company}", styles["TitleBig"]))
    if sub: story.append(Paragraph(sub, styles["Subtitle"]))
    if thesis: story.append(Paragraph(thesis, styles["Body"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Prepared: {date.today().strftime('%B %d, %Y')}", styles["Muted"]))
    story.append(Paragraph("Prepared by: Obin AI Research Analyst", styles["Muted"]))


def add_key_metrics(story, memo):
    km = memo.get("executive_summary", {}).get("key_metrics") or {}
    rows = []
    for label, key, fmt in [
        ("Current Price", "current_price", "currency"),
        ("Market Cap", "market_cap", "currency"),
        ("P/E", "PE", "num"),
        ("EV/EBITDA", "EV_EBITDA", "num"),
        ("P/B", "PB", "num"),
        ("Peer Comparison", "peer_comparison", None),
    ]:
        val = km.get(key)
        if val is not None:
            val = _human_currency(val) if fmt == "currency" else f"{val:.2f}" if fmt == "num" and isinstance(val, (int, float)) else str(val)
            rows.append([label, val])
    if rows:
        story.append(Paragraph("Key Metrics", styles["Section"]))
        tbl = Table([["Metric", "Value"]] + rows, colWidths=[2*inch, 3.6*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EFF3F7")),
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#C8CCD4")),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.1*inch))

def add_financials(story, memo):
    fa = memo.get("financial_analysis", {}) or {}
    if any(fa.values()):
        story.append(Paragraph("Financial Analysis", styles["Section"]))
        for k, title in [
            ("revenue_trends","Revenue Trends"), 
            ("profitability","Profitability"),
            ("cash_flow","Cash Flow"),
            ("balance_sheet","Balance Sheet"),
        ]:
            txt = fa.get(k)
            if txt:
                story.append(Paragraph(title, styles["SectionSmall"]))
                story.append(Paragraph(str(txt), styles["Body"]))

def add_news(story, memo):
    news = memo.get("company_news") or {}
    summary = news.get("summary")
    items = news.get("news_items", [])
    if summary or items:
        story.append(Paragraph("News & Market Position", styles["Section"]))
        if summary:
            story.append(Paragraph(summary, styles["Body"]))
        for item in items[:3]:
            t = item.get("title","")
            c = item.get("content","")
            story.append(Paragraph(f"<b>{t}</b>", styles["Body"]))
            story.append(Paragraph(c, styles["Body"]))

def add_bullets(story, memo, key, title):
    vals = memo.get(key, [])
    if vals:
        story.append(Paragraph(title, styles["Section"]))
        for i, val in enumerate(vals, 1):
            story.append(Paragraph(f"{i}. {val}", styles["CustomBullet"]))

def add_scope(story, memo):
    sc = memo.get("analysis_scope")
    if sc: story.append(Paragraph(f"Analysis Scope: {sc}", styles["Muted"]))

def create_pdf(memo: dict, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []
    add_cover(story, memo)
    add_key_metrics(story, memo)
    add_financials(story, memo)
    add_news(story, memo)
    add_bullets(story, memo, "risks", "Risks")
    add_bullets(story, memo, "catalysts", "Catalysts")
    add_scope(story, memo)
    doc.build(story)

# --- DEMO ---
if __name__ == "__main__":
    demo = {
        "executive_summary": {
            "company": "Example Corp (EXMPL)",
            "recommendation": "BUY",
            "target_price": "$150",
            "time_horizon": "12 months",
            "thesis": "Example Corp is well positioned in a growing market with improving margins and a strong balance sheet.",
            "key_metrics": {
                "current_price": 123.45,
                "market_cap": 12_345_678_900,
                "PE": 18.2,
                "EV_EBITDA": 9.3,
                "PB": 3.1,
                "peer_comparison": "Software — Tech"
            }
        },
        "financial_analysis": {
            "revenue_trends": "Revenue grew 15% YoY driven by higher product sales and recurring subscriptions.",
            "profitability": "Gross margin expanded to 58% from 52% last year.",
            "cash_flow": "Operating cash flow improved to $450M, reflecting better collections.",
            "balance_sheet": "Strong net cash position, low leverage."
        },
        "company_news": {
            "summary": "Recent partnership announced with a major cloud provider to expand distribution.",
            "news_items": [
                {"title": "Partnership announced", "content": "Example Corp announced a strategic partnership.", "url": ""},
                {"title": "Product launch", "content": "New AI-powered module launched with early enterprise adoption.", "url": ""}
            ]
        },
        "risks": ["Execution risk on product rollout", "Supply chain constraints"],
        "catalysts": ["New product release", "International expansion"],
        "analysis_scope": "Comprehensive analysis using filings and market data"
    }
    out = os.path.join(os.getcwd(), "investment_memo_simple.pdf")
    create_pdf(demo, out)
    print(f"PDF generated: {out}")
