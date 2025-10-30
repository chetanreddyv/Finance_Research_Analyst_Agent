"""
================================================================================
SEC EDGAR DOCUMENT PROCESSOR - RAG-OPTIMIZED FOR FINANCIAL ANALYST AGENT MVP
================================================================================

Fixed version with proper HTML document handling (no page_no attribute)
"""

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import time
from datetime import datetime
import re
import tempfile
import shutil


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SECFinancialRAGProcessor:
    """
    Process SEC EDGAR filings for financial research analyst RAG agents.
    Outputs only RAG-ready chunks with metadata - no redundant markdown/JSON exports.
    
    Fixed: Proper handling of HTML documents (no page_no attribute)
    """
    
    def __init__(self, 
                 max_workers: int = 4, 
                 output_dir: str = "rag_knowledge_base",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 512,
                 process_only_main_filing: bool = True):
        """
        Initialize SEC RAG processor with Docling components.
        
        Args:
            max_workers: Parallel processing threads
            output_dir: Output directory for RAG-ready chunks
            embedding_model: HuggingFace model ID for tokenization
            max_tokens: Max tokens per chunk (align with embedding model)
            process_only_main_filing: If True, skip exhibits (recommended for MVP)
        """
        # ====================================================================
        # STEP 1: Initialize Docling DocumentConverter
        # ====================================================================
        self.converter = DocumentConverter()
        logger.info(f"✓ Initialized Docling DocumentConverter")
        
        # ====================================================================
        # STEP 2: Initialize Docling HybridChunker
        # ====================================================================
        # Per Docling docs: HybridChunker uses tokenizer to ensure chunks
        # fit within model input limits
        self.chunker = HybridChunker(
            tokenizer=embedding_model,
            max_tokens=max_tokens,
            merge_peers=True  # Merge small adjacent chunks
        )
        logger.info(
            f"✓ Initialized Docling HybridChunker "
            f"(model: {embedding_model}, max_tokens: {max_tokens})"
        )
        
        # ====================================================================
        # STEP 3: Setup output and processing configuration
        # ====================================================================
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sec_rag_"))
        logger.info(f"✓ Created temporary directory: {self.temp_dir}")
        
        self.max_workers = max_workers
        self.process_only_main = process_only_main_filing
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.results = []
        
        logger.info(
            f"✓ Processor initialized - Output: {self.output_dir}, "
            f"Workers: {max_workers}"
        )
    
    def __del__(self):
        """Cleanup: Remove temporary directory on object destruction"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
    
    # ========================================================================
    # DOCUMENT DISCOVERY AND METADATA EXTRACTION
    # ========================================================================
    
    def find_sec_documents(self, base_path: str = "sec-edgar-filings") -> List[Path]:
        """
        Discover all SEC full-submission.txt files.
        
        Expected structure:
            sec-edgar-filings/{TICKER}/{FORM_TYPE}/{ACCESSION}/full-submission.txt
        """
        base = Path(base_path)
        
        if not base.exists():
            logger.error(f"Base path does not exist: {base_path}")
            return []
        
        documents = list(base.glob("**/full-submission.txt"))
        logger.info(f"📂 Found {len(documents)} SEC filing documents")
        
        return documents
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """
        Extract filing metadata from directory structure.
        
        Path: .../TICKER/FORM_TYPE/ACCESSION_NUMBER/full-submission.txt
        """
        parts = file_path.parts
        
        try:
            metadata = {
                "ticker": parts[-4],
                "form_type": parts[-3],
                "accession_number": parts[-2],
            }
            
            # Extract filing date from accession number
            accession = metadata["accession_number"]
            if '-' in accession:
                metadata["filing_date"] = accession.split('-')[0]
            else:
                metadata["filing_date"] = "UNKNOWN"
                
            return metadata
            
        except IndexError:
            logger.warning(f"⚠️  Could not extract metadata from: {file_path}")
            return {
                "ticker": "UNKNOWN",
                "form_type": "UNKNOWN",
                "accession_number": file_path.stem,
                "filing_date": "UNKNOWN"
            }
    
    # ========================================================================
    # SEC FILING EXTRACTION AND PREPROCESSING
    # ========================================================================
    
    def extract_main_document(self, file_path: Path) -> Dict:
        """
        Extract the main filing document from full-submission.txt.
        
        SEC structure:
            <DOCUMENT> ← Main filing (we want this)
                <TYPE>10-K</TYPE>
                <HTML>...content...</HTML>
            </DOCUMENT>
            <DOCUMENT> ← Exhibit 1 (skip for MVP)
            </DOCUMENT>
        
        Returns:
            Dict with 'type' (filing type) and 'content' (HTML string)
        """
        logger.debug(f"📄 Reading: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract all <DOCUMENT> blocks
        doc_pattern = r'<DOCUMENT>(.*?)</DOCUMENT>'
        documents = re.findall(doc_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not documents:
            raise ValueError(f"No <DOCUMENT> blocks found in {file_path.name}")
        
        logger.debug(f"   Found {len(documents)} document blocks")
        
        # Extract main document (first block only for MVP)
        main_doc = documents[0]
        
        # Extract document type
        type_match = re.search(r'<TYPE>(.*?)\n', main_doc, re.IGNORECASE)
        doc_type = type_match.group(1).strip() if type_match else "UNKNOWN"
        
        # Extract HTML content
        html_pattern = r'<HTML>(.*?)</HTML>'
        html_match = re.search(html_pattern, main_doc, re.DOTALL | re.IGNORECASE)
        
        if html_match:
            html_content = f"<HTML>{html_match.group(1)}</HTML>"
        else:
            logger.debug(f"   No <HTML> tags, using full document")
            html_content = main_doc
        
        # Clean SEC-specific artifacts
        html_content = self._clean_sec_artifacts(html_content)
        
        logger.debug(f"   ✓ Extracted: {doc_type} ({len(html_content):,} chars)")
        
        return {
            "type": doc_type,
            "content": html_content
        }
    
    def _clean_sec_artifacts(self, html_content: str) -> str:
        """
        Remove SEC-specific XML tags that aren't part of HTML standard.
        """
        html_content = html_content.replace('&nbsp;', ' ')
        
        sec_tags = [
            'DOCUMENT', 'TYPE', 'SEQUENCE', 'FILENAME', 
            'DESCRIPTION', 'TEXT', 'SEC-HEADER', 'SEC-DOCUMENT'
        ]
        
        for tag in sec_tags:
            html_content = re.sub(
                f'</?{tag}[^>]*>', 
                '', 
                html_content, 
                flags=re.IGNORECASE
            )
        
        return html_content
    
    # ========================================================================
    # HELPER: Extract Page Numbers from Provenance (HTML-compatible)
    # ========================================================================
    
    def _extract_page_numbers(self, chunk) -> List[int]:
        """
        Extract page numbers from chunk provenance metadata.
        
        For HTML documents, page numbers may not exist. This method safely
        extracts page numbers from the provenance (prov) array if available.
        
        Args:
            chunk: Docling chunk object
            
        Returns:
            List of unique page numbers, or empty list if none found
        """
        page_numbers = set()
        
        try:
            # Check if chunk has doc_items with provenance
            if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    # Check if item has provenance array
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            # Safely get page_no attribute
                            if hasattr(prov, 'page_no') and prov.page_no is not None:
                                page_numbers.add(prov.page_no)
        except Exception as e:
            logger.debug(f"   Could not extract page numbers: {e}")
        
        return sorted(list(page_numbers))
    
    # ========================================================================
    # DOCLING CONVERSION AND CHUNKING
    # ========================================================================
    
    def process_single_document(self, file_path: Path) -> Dict:
        """
        Process a single SEC filing through the complete RAG pipeline.
        
        PIPELINE STAGES:
        1. Metadata Extraction    → Get ticker, form type, filing date
        2. Document Extraction    → Extract main filing
        3. Docling Conversion     → Convert HTML to DoclingDocument
        4. Hybrid Chunking        → Create token-aware chunks
        5. Chunk Contextualization→ Add header context
        6. Metadata Enrichment    → Add filing and chunk metadata
        7. Output Generation      → Save as JSONL
        """
        start_time = time.time()
        
        try:
            # ================================================================
            # STAGE 1: Extract Metadata
            # ================================================================
            metadata = self.extract_metadata_from_path(file_path)
            
            logger.info(
                f"🔄 Processing: {metadata['ticker']} | "
                f"{metadata['form_type']} | {metadata['accession_number']}"
            )
            
            # ================================================================
            # STAGE 2: Extract Main Document
            # ================================================================
            logger.debug("   Stage 1/5: Extracting main document...")
            doc_block = self.extract_main_document(file_path)
            
            # ================================================================
            # STAGE 3: Save to Temp HTML
            # ================================================================
            logger.debug("   Stage 2/5: Preparing for Docling...")
            
            temp_filename = (
                f"{metadata['ticker']}_{metadata['form_type']}_"
                f"{metadata['accession_number']}.html"
            )
            temp_html = self.temp_dir / temp_filename
            
            with open(temp_html, 'w', encoding='utf-8') as f:
                f.write(doc_block['content'])
            
            logger.debug(f"   Temp file: {temp_html.name}")
            
            # ================================================================
            # STAGE 4: Convert with Docling
            # ================================================================
            logger.debug("   Stage 3/5: Converting with Docling...")
            
            result = self.converter.convert(str(temp_html))
            docling_document = result.document
            
            logger.debug(
                f"   ✓ Conversion complete: {len(docling_document.pages)} pages"
            )
            
            # ================================================================
            # STAGE 5: Chunk with HybridChunker
            # ================================================================
            logger.debug("   Stage 4/5: Chunking with HybridChunker...")
            
            chunks = []
            
            for chunk_idx, chunk in enumerate(self.chunker.chunk(docling_document)):
                
                # ============================================================
                # STAGE 5a: Contextualize Chunk
                # ============================================================
                # Per Docling docs: contextualize() adds header context
                contextualized_text = self.chunker.contextualize(chunk)
                
                # ============================================================
                # STAGE 5b: Extract Page Numbers (HTML-safe)
                # ============================================================
                # For HTML documents, page numbers may not exist
                # Use provenance metadata if available
                page_numbers = self._extract_page_numbers(chunk)
                primary_page = page_numbers[0] if page_numbers else None
                
                # ============================================================
                # STAGE 5c: Extract Chunk Metadata
                # ============================================================
                chunk_metadata = {
                    # --- Chunk Identifiers ---
                    "chunk_id": chunk_idx,
                    "chunk_index": chunk_idx,
                    
                    # --- Document-Level Metadata (for filtering) ---
                    "ticker": metadata['ticker'],
                    "company": metadata['ticker'],
                    "form_type": metadata['form_type'],
                    "filing_type": metadata['form_type'],
                    "accession_number": metadata['accession_number'],
                    "filing_date": metadata['filing_date'],
                    
                    # --- Chunk-Level Structural Metadata ---
                    # Page numbers (HTML-safe extraction from provenance)
                    "page": primary_page,  # First page number or None
                    "page_numbers": page_numbers,  # All page numbers
                    
                    # Section headings (for context and filtering)
                    "headings": (
                        chunk.meta.headings 
                        if hasattr(chunk.meta, 'headings') 
                        else []
                    ),
                    
                    # Primary section (top-level heading)
                    "section": (
                        chunk.meta.headings[0] 
                        if hasattr(chunk.meta, 'headings') and chunk.meta.headings 
                        else None
                    ),
                    
                    # --- Document Item Types (for filtering) ---
                    "doc_items": (
                        [item.label for item in chunk.meta.doc_items]
                        if hasattr(chunk.meta, 'doc_items')
                        else []
                    ),
                    
                    # Boolean flag for quick table filtering
                    "has_table": (
                        any(item.label == "table" for item in chunk.meta.doc_items)
                        if hasattr(chunk.meta, 'doc_items')
                        else False
                    ),
                    
                    # --- Quality Metrics ---
                    "token_count": len(contextualized_text.split()),
                    "char_count": len(contextualized_text),
                }
                
                # ============================================================
                # STAGE 5d: Create Chunk Object
                # ============================================================
                chunk_data = {
                    "text": contextualized_text,  # ← Gets embedded
                    "metadata": chunk_metadata,   # ← Enables filtering
                }
                
                chunks.append(chunk_data)
            
            # ================================================================
            # STAGE 6: Calculate Statistics
            # ================================================================
            processing_time = time.time() - start_time
            
            avg_tokens = (
                sum(c['metadata']['token_count'] for c in chunks) / len(chunks)
                if chunks 
                else 0
            )
            
            processing_result = {
                "status": "success",
                "metadata": metadata,
                "processing_time_seconds": round(processing_time, 2),
                "total_chunks": len(chunks),
                "total_pages": len(docling_document.pages),
                "avg_tokens_per_chunk": round(avg_tokens, 1),
                "chunks": chunks,
                "processed_at": datetime.now().isoformat()
            }
            
            # ================================================================
            # STAGE 7: Save RAG-Ready Chunks
            # ================================================================
            logger.debug("   Stage 5/5: Saving RAG-ready chunks...")
            self._save_rag_chunks(processing_result)
            
            # ================================================================
            # Log Success
            # ================================================================
            logger.info(
                f"✅ SUCCESS: {metadata['ticker']} {metadata['form_type']} → "
                f"{len(chunks)} chunks, {len(docling_document.pages)} pages, "
                f"{processing_time:.2f}s"
            )
            
            return processing_result
            
        except Exception as e:
            # ================================================================
            # Handle Errors Gracefully
            # ================================================================
            processing_time = time.time() - start_time
            
            error_result = {
                "status": "failed",
                "metadata": self.extract_metadata_from_path(file_path),
                "processing_time_seconds": round(processing_time, 2),
                "error": str(e),
                "error_type": type(e).__name__,
                "error_details": {
                    "file_path": str(file_path),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.error(
                f"❌ FAILED: {file_path.name} - "
                f"{type(e).__name__}: {str(e)}"
            )
            
            return error_result
    
    # ========================================================================
    # OUTPUT AND STORAGE
    # ========================================================================
    
    def _save_rag_chunks(self, result: Dict):
        """
        Save RAG-ready chunks to JSONL format.
        
        Output: {TICKER}/{FORM_TYPE}/{ACCESSION}_chunks.jsonl
        """
        metadata = result['metadata']
        
        # Create output directory structure
        output_subdir = (
            self.output_dir / 
            metadata['ticker'] / 
            metadata['form_type']
        )
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks as JSONL
        chunks_file = (
            output_subdir / 
            f"{metadata['accession_number']}_chunks.jsonl"
        )
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in result.get('chunks', []):
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        logger.debug(f"   💾 Saved {result['total_chunks']} chunks")
        
        # Save processing summary
        summary = {
            "ticker": metadata['ticker'],
            "form_type": metadata['form_type'],
            "accession_number": metadata['accession_number'],
            "filing_date": metadata['filing_date'],
            "total_chunks": result['total_chunks'],
            "total_pages": result['total_pages'],
            "avg_tokens_per_chunk": result['avg_tokens_per_chunk'],
            "processing_time_seconds": result['processing_time_seconds'],
            "processed_at": result['processed_at'],
            "chunks_file": str(chunks_file.relative_to(self.output_dir)),
            "embedding_model": self.embedding_model,
            "max_tokens": self.max_tokens,
        }
        
        summary_file = (
            output_subdir / 
            f"{metadata['accession_number']}_summary.json"
        )
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def batch_process(
        self, 
        documents: List[Path] = None, 
        base_path: str = "sec-edgar-filings"
    ) -> Dict:
        """
        Process multiple SEC filings in parallel.
        """
        if documents is None:
            documents = self.find_sec_documents(base_path)
        
        if not documents:
            logger.warning("⚠️  No documents found")
            return {
                "total_documents": 0,
                "successful": 0,
                "failed": 0
            }
        
        logger.info(
            f"\n{'='*80}\n"
            f"  RAG KNOWLEDGE BASE CREATION FOR FINANCIAL ANALYST AGENT\n"
            f"{'='*80}\n"
            f"  Documents to process: {len(documents)}\n"
            f"  Parallel workers: {self.max_workers}\n"
            f"  Embedding model: {self.embedding_model}\n"
            f"  Max tokens per chunk: {self.max_tokens}\n"
            f"  Process main filing only: {self.process_only_main}\n"
            f"  Output directory: {self.output_dir}\n"
            f"{'='*80}"
        )
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self.process_single_document, doc): doc
                for doc in documents
            }
            
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Executor exception: {doc.name} - {e}")
                    results.append({
                        "status": "failed",
                        "metadata": self.extract_metadata_from_path(doc),
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
        
        total_time = time.time() - start_time
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        total_chunks = sum(r.get('total_chunks', 0) for r in successful_results)
        total_pages = sum(r.get('total_pages', 0) for r in successful_results)
        
        summary = {
            "total_documents": len(documents),
            "successful": success_count,
            "failed": failed_count,
            "total_processing_time_seconds": round(total_time, 2),
            "average_time_per_doc": (
                round(total_time / len(documents), 2) 
                if documents 
                else 0
            ),
            "knowledge_base_stats": {
                "total_chunks_created": total_chunks,
                "total_pages_processed": total_pages,
                "avg_chunks_per_filing": (
                    round(total_chunks / success_count, 1) 
                    if success_count > 0 
                    else 0
                ),
                "ready_for_vector_db": True,
                "embedding_model": self.embedding_model,
                "max_tokens_per_chunk": self.max_tokens,
            },
            "results": [
                {k: v for k, v in r.items() if k != 'chunks'}
                for r in results
            ]
        }
        
        # Save batch summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = self.output_dir / f"batch_summary_{timestamp}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Create master index
        self._create_master_index(results)
        
        logger.info(
            f"\n{'='*80}\n"
            f"  ✅ RAG KNOWLEDGE BASE CREATED!\n"
            f"{'='*80}\n"
            f"  📊 SUMMARY:\n"
            f"     Total: {len(documents)} | Success: {success_count} | "
            f"Failed: {failed_count}\n"
            f"     Total Chunks: {total_chunks:,}\n"
            f"     Total Pages: {total_pages:,}\n"
            f"     Total Time: {total_time:.2f}s\n"
            f"     Avg Time/Doc: {summary['average_time_per_doc']:.2f}s\n"
            f"\n"
            f"  📂 OUTPUT:\n"
            f"     Directory: {self.output_dir}\n"
            f"     Summary: {summary_path.name}\n"
            f"{'='*80}"
        )
        
        self.results = results
        return summary
    
    def _create_master_index(self, results: List[Dict]):
        """
        Create a master index of all successfully processed documents.
        """
        index_entries = []
        
        for result in results:
            if result['status'] == 'success':
                meta = result['metadata']
                
                index_entries.append({
                    "ticker": meta['ticker'],
                    "form_type": meta['form_type'],
                    "accession_number": meta['accession_number'],
                    "filing_date": meta['filing_date'],
                    "chunks_count": result['total_chunks'],
                    "pages_count": result['total_pages'],
                    "avg_tokens_per_chunk": result['avg_tokens_per_chunk'],
                    "chunks_file": (
                        f"{meta['ticker']}/{meta['form_type']}/"
                        f"{meta['accession_number']}_chunks.jsonl"
                    ),
                    "processed_at": result['processed_at'],
                    "processing_time_seconds": result['processing_time_seconds'],
                })
        
        master_index = {
            "total_filings": len(index_entries),
            "total_chunks": sum(e['chunks_count'] for e in index_entries),
            "total_pages": sum(e['pages_count'] for e in index_entries),
            "created_at": datetime.now().isoformat(),
            "embedding_model": self.embedding_model,
            "max_tokens_per_chunk": self.max_tokens,
            "filings": index_entries
        }
        
        index_path = self.output_dir / "master_index.json"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(master_index, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   📇 Master index created: {index_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main entry point for SEC RAG knowledge base creation.
    
    FIXED: Proper HTML document handling (no page_no attribute errors)
    """
    print(
        "\n"
        "╔════════════════════════════════════════════════════════════════════╗\n"
        "║  SEC EDGAR RAG PROCESSOR - FINANCIAL ANALYST AGENT MVP            ║\n"
        "║  Fixed: HTML document handling (no page_no errors)                ║\n"
        "╚════════════════════════════════════════════════════════════════════╝\n"
    )
    
    processor = SECFinancialRAGProcessor(
        output_dir="rag_knowledge_base",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=512,
        max_workers=4,
        process_only_main_filing=True,
    )
    
    summary = processor.batch_process(base_path="sec-edgar-filings")
    
    print(
        f"\n"
        f"╔════════════════════════════════════════════════════════════════════╗\n"
        f"║  🎯 RAG KNOWLEDGE BASE READY!                                     ║\n"
        f"╚════════════════════════════════════════════════════════════════════╝\n"
        f"\n"
        f"  📊 STATS:\n"
        f"     Total Chunks: {summary['knowledge_base_stats']['total_chunks_created']:,}\n"
        f"     Avg Chunks/Filing: {summary['knowledge_base_stats']['avg_chunks_per_filing']}\n"
        f"     Success Rate: {(summary['successful']/summary['total_documents']*100):.1f}%\n"
        f"\n"
        f"  📂 OUTPUT: {processor.output_dir}/\n"
        f"\n"
        f"  🚀 NEXT: Load JSONL files into vector database\n"
    )


if __name__ == "__main__":
    main()
