┌─────────────────────────────────────────────────────────────────┐
│              Unstructured Documents Input                       │
│    (PDFs, DOCX, PPTX, Images, Scanned Documents, 10-K/10-Q)     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DOCLING                                    │
│    • Advanced PDF Understanding (No OCR for native PDFs)        │
│    • Vision Models for Layout Recognition                       │
│    • TableFormer for Table Extraction                           │
│    • 30x Faster than Traditional OCR                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│               Docling Documents (Structured)                     │
│         JSON, Markdown with Semantic Structure                   │
│    • Classified Text Blocks (Titles, Paragraphs, Tables)       │
│    • Extracted Tables (Rows/Columns)                            │
│    • Images & Figures with Captions                             │
│    • Document Metadata                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Database                               │
│              (Pinecone / Chroma / Weaviate)                     │
│    • Semantic Embeddings of Document Chunks                     │
│    • Efficient Similarity Search                                │
│    • Metadata Filtering (Company, Date, Document Type)         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RAG (Retrieval-Augmented Generation)            │
│    • Query → Embedding → Vector Search                          │
│    • Retrieve Relevant Document Chunks                          │
│    • Context-Aware LLM Prompting                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Processing Layer                          │
│              (GPT-4, Claude 3.5, Gemini 2.0)                    │
│                   + FastMCP Tools                                │
│    • Web Search Tool (Real-time Data)                           │
│    • News API Tool (Market Sentiment)                           │
│    • SEC EDGAR Tool (Latest Filings)                            │
│    • Calculation Tool (Financial Ratios)                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│          Analysis & Report Generation Agent                      │
│                   (LLM Orchestration)                            │
│    • Investment Memos                                            │
│    • Risk Reports & Red Flag Detection                          │
│    • Financial Health Summaries                                 │
│    • Competitor Benchmarking                                     │
└─────────────────────────────────────────────────────────────────┘
