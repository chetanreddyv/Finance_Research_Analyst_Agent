"""
End-to-End SEC EDGAR RAG Chunk Loader for Pinecone
Handles all chunks, prevents vector overwrites, supports any ticker/form type.
Tested for AAPL 8-K, 10-K, 10-Q filings as example.
"""

import json
from pathlib import Path
import os
import time
import uuid
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
PINECONE_API_KEY = "xxxxxxxxxxxxx"


INDEX_NAME = "sec-rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = "rag_knowledge_base"

# ============================================================================
# SETUP
# ============================================================================
print("🔧 Setting up...")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete any old index if starting fresh (comment if you want to append)
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME in existing_indexes:
    print(f"Deleting existing index {INDEX_NAME} for fresh ingest")
    pc.delete_index(INDEX_NAME)
    time.sleep(5)

# Create index
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Waiting for index to be ready...")
    time.sleep(10)

# Connect to index
index = pc.Index(INDEX_NAME)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Setup complete!\n")

# ============================================================================
# LOAD CHUNKS
# ============================================================================
print("📂 Finding chunk files...")
chunk_files = list(Path(KNOWLEDGE_BASE_PATH).glob("**/*_chunks.jsonl"))
print(f"Found {len(chunk_files)} files\n")

if not chunk_files:
    print("❌ No chunk files found! Check your KNOWLEDGE_BASE_PATH.")
    exit(1)

total_chunks_processed = 0
total_vectors_upserted = 0
batch_size = 100

for file_path in tqdm(chunk_files, desc="Processing files"):
    # Read all chunks from JSONL
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Batch upserts for efficiency
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Prepare vectors for Pinecone, with unique IDs to prevent overwrites
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            meta = chunk["metadata"]
            unique_suffix = uuid.uuid4().hex[:8]
            # Unique ID combines ticker, form type, accession, chunk index, and uuid
            vector_id = (
                f"{meta.get('ticker', 'UNK')}_"
                f"{meta.get('form_type', 'UNK')}_"
                f"{meta.get('accession_number', 'UNK')}_"
                f"{meta.get('chunk_index', 0)}_{unique_suffix}"
            )
            vector_metadata = {
                "text": chunk["text"][:1000],  # Truncate to 1000 chars
                "ticker": meta.get("ticker", ""),
                "form_type": meta.get("form_type", ""),
                "accession_number": meta.get("accession_number", ""),
                "filing_date": meta.get("filing_date", ""),
                "section": meta.get("section", "") or "",
                "has_table": meta.get("has_table", False),
                "chunk_index": meta.get("chunk_index", 0),
            }
            vectors.append({
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": vector_metadata
            })
        
        # Upsert to Pinecone
        try:
            index.upsert(vectors=vectors)
            total_vectors_upserted += len(vectors)
            total_chunks_processed += len(batch)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to upsert batch for {file_path}: {e}")

print(f"\n✅ Done! Processed {total_chunks_processed:,} chunks, upserted {total_vectors_upserted:,} vectors into Pinecone.")

# Wait for final indexing
print("Waiting for Pinecone to finish indexing...")
time.sleep(5)
stats = index.describe_index_stats()
print(f"📊 Final Index Stats:")
print(f"   Total vectors: {stats.total_vector_count:,}")
print(f"   Dimension: {stats.dimension}")
print(f"   Data Integrity: {stats.total_vector_count / total_chunks_processed:.2%} match\n")

# ============================================================================
# SAMPLE QUERY
# ============================================================================
print("🔍 Testing with sample queries...\n")

sample_queries = [
    ("Apple Inc total revenue fiscal year 2024", {"ticker": "AAPL", "form_type": "10-K"}),
    ("Apple quarterly report Q2 net sales", {"ticker": "AAPL", "form_type": "10-Q"}),
    ("Apple 8-K filing risk factors", {"ticker": "AAPL", "form_type": "8-K"}),
]

for q, fltr in sample_queries:
    print(f"Query: {q}, Filter: {fltr}")
    query_embedding = model.encode([q])[0].tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True, filter=fltr)
    for i, match in enumerate(results.matches, 1):
        score_emoji = "🟢" if match.score > 0.7 else "🟡" if match.score > 0.6 else "🔴"
        print(f"{i}. {score_emoji} Score: {match.score:.4f}")
        print(f"   {match.metadata.get('text', '')[:120]}...\n")
    print('-' * 70 + "\n")

print("🎉 All AAPL 8-K, 10-K, and 10-Q chunks in Pinecone and ready for RAG!")
