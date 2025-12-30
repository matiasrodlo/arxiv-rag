"""
Main RAG Pipeline
Orchestrates the complete pipeline from PDF extraction to vector store indexing.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

from ..extractors.pdf_extractor import PDFExtractor, load_metadata
from ..processors.text_processor import TextProcessor, TextChunker
from ..embeddings.embedder import Embedder
from ..storage.vector_store import VectorStore

# Import worker function for multiprocessing
try:
    from .worker import _process_paper_worker
except ImportError:
    # Fallback: define worker function here if import fails
    _process_paper_worker = None


class RAGPipeline:
    """Complete RAG pipeline for processing ArXiv papers."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(
            enable_ocr=self.config['pdf_extraction']['enable_ocr'],
            ocr_language=self.config['pdf_extraction']['ocr_language']
        )
        
        self.text_processor = TextProcessor(
            remove_headers_footers=self.config['text_processing']['remove_headers_footers'],
            normalize_whitespace=self.config['text_processing']['normalize_whitespace'],
            fix_encoding=self.config['text_processing']['fix_encoding'],
            improve_formulas=self.config['text_processing'].get('improve_formulas', True)
        )
        
        self.chunker = TextChunker(
            method=self.config['chunking']['method'],
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap'],
            model_name=self.config['chunking'].get('model'),
            min_chunk_size=self.config['text_processing']['min_chunk_size'],
            max_chunk_size=self.config['text_processing']['max_chunk_size']
        )
        
        # Initialize embeddings only if needed during processing
        self.embedder = None
        if self.config.get('embeddings', {}).get('generate_during_processing', True):
            try:
                self.embedder = Embedder(
                    model_name=self.config['embeddings']['model'],
                    batch_size=self.config['embeddings']['batch_size'],
                    device=self.config['embeddings']['device'],
                    normalize_embeddings=self.config['embeddings']['normalize_embeddings']
                )
            except ImportError:
                logger.warning("Embeddings not available, will skip embedding generation during processing")
        
        # Initialize vector store only if needed during processing
        self.vector_store = None
        if self.config.get('vector_db', {}).get('add_during_processing', True):
            try:
                self.vector_store = VectorStore(
                    db_type=self.config['vector_db']['type'],
                    collection_name=self.config['vector_db']['collection_name'],
                    persist_directory=self.config['vector_db'].get('persist_directory'),
                    qdrant_host=self.config['vector_db'].get('qdrant_host', 'localhost'),
                    qdrant_port=self.config['vector_db'].get('qdrant_port', 6333)
                )
            except ImportError:
                logger.warning("Vector store not available, will skip vector store addition during processing")
        
        # Setup paths
        self.pdf_dir = Path(self.config['paths']['pdf_dir'])
        self.metadata_dir = Path(self.config['paths']['metadata_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.extracted_text_dir = Path(self.config['paths']['extracted_text_dir']).resolve()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        
        # #region agent log
        log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "pipeline.py:97",
                    "message": "Pipeline init - extracted_text_dir setup",
                    "data": {
                        "config_value": self.config['paths']['extracted_text_dir'],
                        "resolved_path": str(self.extracted_text_dir),
                        "exists": self.extracted_text_dir.exists(),
                        "is_dir": self.extracted_text_dir.is_dir() if self.extracted_text_dir.exists() else False
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Initialize progress tracking database
        self.progress_db_path = Path(self.config['paths'].get('progress_db', 'data/progress.db'))
        self.progress_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_progress_db()
        
        logger.info("RAG Pipeline initialized")
    
    def _init_progress_db(self):
        """Initialize SQLite database for progress tracking."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_papers (
                paper_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_chunks INTEGER,
                text_length INTEGER,
                error TEXT
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status ON processed_papers(status)
        ''')
        conn.commit()
        conn.close()
    
    def _is_processed(self, paper_id: str) -> bool:
        """Check if a paper has already been processed successfully."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT status FROM processed_papers WHERE paper_id = ?', (paper_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0] == 'success'
        
        # Also check if JSON file exists
        json_path = self.extracted_text_dir / f"{paper_id}.json"
        return json_path.exists()
    
    def _mark_processed(self, paper_id: str, status: str, num_chunks: int = 0, text_length: int = 0, error: str = None):
        """Mark a paper as processed in the database."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO processed_papers 
            (paper_id, status, num_chunks, text_length, error)
            VALUES (?, ?, ?, ?, ?)
        ''', (paper_id, status, num_chunks, text_length, error))
        conn.commit()
        conn.close()
    
    def _mark_processed_batch(self, records: List[tuple]):
        """
        Batch mark multiple papers as processed in the database.
        
        Args:
            records: List of tuples (paper_id, status, num_chunks, text_length, error)
        """
        if not records:
            return
        
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO processed_papers 
            (paper_id, status, num_chunks, text_length, error)
            VALUES (?, ?, ?, ?, ?)
        ''', records)
        conn.commit()
        conn.close()
    
    def get_progress_stats(self) -> Dict:
        """Get statistics about processing progress."""
        conn = sqlite3.connect(str(self.progress_db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT status, COUNT(*) FROM processed_papers GROUP BY status')
        results = dict(cursor.fetchall())
        conn.close()
        
        return {
            'success': results.get('success', 0),
            'failed': results.get('failed', 0),
            'total_processed': sum(results.values())
        }
    
    def process_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Process a single paper through the complete pipeline.
        
        Args:
            paper_id: ArXiv paper ID (e.g., "2511.22108v1")
            
        Returns:
            Dictionary with processing results or None if failed
        """
        # Skip macOS resource fork files
        if paper_id.startswith('._'):
            logger.debug(f"Skipping macOS resource fork file: {paper_id}")
            return {
                'paper_id': paper_id,
                'success': False,
                'error': 'macOS resource fork file (not a real PDF)'
            }
        
        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        metadata_path = self.metadata_dir / f"{paper_id}.txt"
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return None
        
        try:
            # Step 1: Extract PDF text
            logger.debug(f"Extracting text from {paper_id}")
            extraction_result = self.pdf_extractor.extract(str(pdf_path))
            
            if not extraction_result['success']:
                logger.error(f"Failed to extract text from {paper_id}: {extraction_result.get('error')}")
                return None
            
            # Step 2: Load metadata
            metadata = load_metadata(str(metadata_path)) if metadata_path.exists() else {}
            metadata['paper_id'] = paper_id
            metadata['extraction_method'] = extraction_result['method_used']
            
            # Step 3: Clean text
            cleaned_text = self.text_processor.clean(extraction_result['text'])
            
            if len(cleaned_text) < self.config['pdf_extraction']['min_text_length']:
                logger.warning(f"Text too short for {paper_id}: {len(cleaned_text)} chars")
                return None
            
            # Step 4: Chunk text
            chunks = self.chunker.chunk(cleaned_text, metadata=metadata)
            
            if not chunks:
                logger.warning(f"No chunks created for {paper_id}")
                return None
            
            # Step 5: Generate embeddings (if embedder is available)
            chunks_with_embeddings = chunks
            if self.embedder:
                try:
                    chunks_with_embeddings = self.embedder.embed_chunks(chunks, show_progress=False)
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}, continuing without embeddings")
                    chunks_with_embeddings = chunks
            
            # Step 6: Add to vector store (if vector store is available)
            if self.vector_store:
                try:
                    self.vector_store.add_chunks(chunks_with_embeddings)
                except Exception as e:
                    logger.warning(f"Vector store addition failed: {e}, continuing")
            
            # Step 7: Extract sections for better structure (with page information)
            pages_data = extraction_result.get('pages', [])
            sections = self.text_processor.extract_sections(cleaned_text, pages=pages_data)
            
            # Step 8: Map chunks to sections for better metadata (optimized with binary search)
            # Pre-build section boundaries for faster lookup
            section_boundaries = [(s['start_char'], s['end_char'], s['name']) for s in sections]
            section_boundaries.sort()  # Sort by start_char
            
            def find_section_for_position(pos: int) -> str:
                """Find which section a character position belongs to (binary search)."""
                left, right = 0, len(section_boundaries) - 1
                while left <= right:
                    mid = (left + right) // 2
                    start, end, name = section_boundaries[mid]
                    if start <= pos < end:
                        return name
                    elif pos < start:
                        right = mid - 1
                    else:
                        left = mid + 1
                return 'Unknown'
            
            # Pre-build page boundaries for faster lookup
            page_boundaries = []
            current_pos = 0
            for page in pages_data:
                page_text = page.get('text', '')
                page_length = len(page_text)
                page_boundaries.append((current_pos, current_pos + page_length, page.get('page', 1)))
                current_pos += page_length + 2  # +2 for page separator
            
            def find_page_for_position(pos: int) -> int:
                """Find which page a character position belongs to (binary search)."""
                left, right = 0, len(page_boundaries) - 1
                while left <= right:
                    mid = (left + right) // 2
                    start, end, page_num = page_boundaries[mid]
                    if start <= pos < end:
                        return page_num
                    elif pos < start:
                        right = mid - 1
                    else:
                        left = mid + 1
                return page_boundaries[-1][2] if page_boundaries else 1
            
            # Step 9: Prepare enhanced JSON structure optimized for RAG
            chunk_start_positions = []
            current_pos = 0
            for chunk in chunks:
                chunk_start_positions.append(current_pos)
                current_pos += len(chunk['text']) + 1  # +1 for separator
            
            extracted_data = {
                'paper_id': paper_id,
                'metadata': {
                    **metadata,
                    'extraction_method': extraction_result['method_used'],
                    'quality_score': extraction_result.get('quality_score', 0.0),
                    'pdf_type': extraction_result.get('pdf_type', 'unknown'),
                    'num_pages': len(pages_data),
                    'text_length': len(cleaned_text),
                    'extraction_date': extraction_result.get('extraction_date', ''),
                },
                'text': {
                    'full': cleaned_text,
                    'by_page': [
                        {
                            'page': p.get('page', i + 1),
                            'text': p.get('text', ''),
                            'char_count': p.get('char_count', len(p.get('text', '')))
                        }
                        for i, p in enumerate(pages_data)
                    ],
                    'sections': [
                        {
                            'name': s.get('name', ''),
                            'text': s.get('text', ''),
                            'start_char': s.get('start_char', 0),
                            'end_char': s.get('end_char', 0),
                            'page': s.get('page', 1)
                        }
                        for s in sections
                    ]
                },
                'chunks': [
                    {
                        'chunk_id': f"{paper_id}_chunk_{i}",
                        'text': chunk['text'],
                        'metadata': {
                            **chunk.get('metadata', {}),
                            'chunk_index': i,
                            'chunk_length': len(chunk['text']),
                            'paper_id': paper_id,
                            'section': find_section_for_position(chunk_start_positions[i]) if i < len(chunk_start_positions) else 'Unknown',
                            'page': find_page_for_position(chunk_start_positions[i]) if i < len(chunk_start_positions) else 1
                        }
                    }
                    for i, chunk in enumerate(chunks_with_embeddings)
                ],
                'statistics': {
                    'num_chunks': len(chunks),
                    'num_pages': len(pages_data),
                    'num_sections': len(sections),
                    'total_chars': len(cleaned_text),
                    'avg_chunk_size': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0,
                    'chunking_method': self.chunker.method
                }
            }
            
            # Save enhanced JSON structure (formatted for readability)
            extracted_text_path = self.extracted_text_dir / f"{paper_id}.json"
            with open(extracted_text_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved enhanced JSON extraction for {paper_id}")
            
            return {
                'paper_id': paper_id,
                'success': True,
                'num_chunks': len(chunks),
                'text_length': len(cleaned_text),
                'extraction_method': extraction_result['method_used']
            }
            
        except Exception as e:
            logger.error(f"Error processing {paper_id}: {e}", exc_info=True)
            return {
                'paper_id': paper_id,
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, paper_ids: List[str], batch_size: Optional[int] = None, 
                      skip_processed: bool = True, num_workers: Optional[int] = None) -> Dict:
        """
        Process a batch of papers with parallel processing.
        
        Args:
            paper_ids: List of paper IDs to process
            batch_size: Optional batch size (from config if not provided)
            skip_processed: Skip papers that are already processed
            num_workers: Number of parallel workers (from config if not provided)
            
        Returns:
            Dictionary with processing statistics
        """
        batch_size = batch_size or self.config['processing']['batch_size']
        num_workers = num_workers or self.config['processing']['num_workers']
        
        # Filter out macOS resource fork files (._*)
        original_count = len(paper_ids)
        paper_ids = [pid for pid in paper_ids if not pid.startswith('._')]
        macos_filtered = original_count - len(paper_ids)
        if macos_filtered > 0:
            logger.info(f"Filtered out {macos_filtered} macOS resource fork files")
        
        # Filter out already processed papers
        skipped = 0
        if skip_processed:
            original_count = len(paper_ids)
            logger.info(f"Checking {original_count} papers for already processed status...")
            # #region agent log
            log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "I",
                        "location": "pipeline.py:424",
                        "message": "Starting skip_processed check",
                        "data": {
                            "total_papers": original_count,
                            "skip_processed": skip_processed
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            
            # Batch check for processed papers (more efficient)
            import sqlite3
            conn = sqlite3.connect(str(self.progress_db_path))
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(paper_ids))
            cursor.execute(f'SELECT paper_id FROM processed_papers WHERE paper_id IN ({placeholders}) AND status="success"', paper_ids)
            processed_set = set(row[0] for row in cursor.fetchall())
            conn.close()
            
            # Also check for existing JSON files
            existing_files = {f.stem for f in self.extracted_text_dir.glob("*.json") if not f.name.startswith('._')}
            processed_set.update(existing_files)
            
            paper_ids = [pid for pid in paper_ids if pid not in processed_set]
            skipped = original_count - len(paper_ids)
            
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "I",
                        "location": "pipeline.py:450",
                        "message": "Skip check completed",
                        "data": {
                            "skipped": skipped,
                            "remaining": len(paper_ids)
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            
            if skipped > 0:
                logger.info(f"Skipping {skipped} already processed papers")
        
        if not paper_ids:
            logger.info("All papers already processed")
            return {
                'total': original_count if skip_processed else 0,
                'successful': skipped,
                'failed': 0,
                'skipped': skipped,
                'errors': []
            }
        
        results = {
            'total': len(paper_ids),
            'successful': 0,
            'failed': 0,
            'skipped': skipped,
            'errors': []
        }
        
        # Process in batches with parallel workers
        total_batches = (len(paper_ids) + batch_size - 1) // batch_size
        
        with tqdm(total=len(paper_ids), desc="Processing papers") as pbar:
            for batch_idx in range(0, len(paper_ids), batch_size):
                batch = paper_ids[batch_idx:batch_idx + batch_size]
                
                # Process batch in parallel
                if _process_paper_worker is None:
                    # Fallback to sequential processing if worker not available
                    logger.warning("Worker function not available, using sequential processing")
                    batch_db_records = []
                    batch_size_db = 50  # Write to DB every 50 papers
                    
                    for paper_id in batch:
                        result = self.process_paper(paper_id)
                        if result and result.get('success'):
                            results['successful'] += 1
                            batch_db_records.append((
                                paper_id, 'success',
                                result.get('num_chunks', 0),
                                result.get('text_length', 0),
                                None
                            ))
                        else:
                            results['failed'] += 1
                            error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
                            batch_db_records.append((
                                paper_id, 'failed', 0, 0, error_msg
                            ))
                            results['errors'].append({
                                'paper_id': paper_id,
                                'error': error_msg
                            })
                        pbar.update(1)
                        
                        # Batch write to database
                        if len(batch_db_records) >= batch_size_db:
                            self._mark_processed_batch(batch_db_records)
                            batch_db_records = []
                    
                    # Write remaining records
                    if batch_db_records:
                        self._mark_processed_batch(batch_db_records)
                else:
                    # Batch database writes for better performance
                    batch_db_records = []
                    batch_size_db = 50  # Write to DB every 50 papers
                    
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        # Submit all papers in batch
                        # #region agent log
                        log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
                        try:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                import json as json_lib
                                log_entry = {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H",
                                    "location": "pipeline.py:477",
                                    "message": "Submitting worker tasks",
                                    "data": {
                                        "batch_size": len(batch),
                                        "extracted_text_dir": str(self.extracted_text_dir),
                                        "extracted_text_dir_resolved": str(self.extracted_text_dir.resolve()),
                                        "cwd": os.getcwd()
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }
                                log_file.write(json_lib.dumps(log_entry) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        future_to_paper = {
                            executor.submit(_process_paper_worker, str(self.pdf_dir.resolve()), 
                                           str(self.metadata_dir.resolve()), str(self.extracted_text_dir.resolve()),
                                           str(self.progress_db_path.resolve()), paper_id, self.config): paper_id
                            for paper_id in batch
                        }
                    
                        # Collect results as they complete
                        for future in as_completed(future_to_paper):
                            paper_id = future_to_paper[future]
                            try:
                                result = future.result()
                                if result and result.get('success'):
                                    results['successful'] += 1
                                    batch_db_records.append((
                                        paper_id, 'success',
                                        result.get('num_chunks', 0),
                                        result.get('text_length', 0),
                                        None
                                    ))
                                else:
                                    results['failed'] += 1
                                    error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
                                    batch_db_records.append((
                                        paper_id, 'failed', 0, 0, error_msg
                                    ))
                                    results['errors'].append({
                                        'paper_id': paper_id,
                                        'error': error_msg
                                    })
                            except Exception as e:
                                results['failed'] += 1
                                error_msg = str(e)
                                batch_db_records.append((
                                    paper_id, 'failed', 0, 0, error_msg
                                ))
                                results['errors'].append({
                                    'paper_id': paper_id,
                                    'error': error_msg
                                })
                            finally:
                                pbar.update(1)
                                
                                # Batch write to database
                                if len(batch_db_records) >= batch_size_db:
                                    self._mark_processed_batch(batch_db_records)
                                    batch_db_records = []
                    
                    # Write remaining records
                    if batch_db_records:
                        self._mark_processed_batch(batch_db_records)
        
        logger.info(f"Batch processing complete: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def process_all(self, paper_ids_file: str = "arxiv_cs_papers/downloaded_ids.txt") -> Dict:
        """
        Process all papers from the IDs file.
        
        Args:
            paper_ids_file: Path to file containing paper IDs (one per line)
            
        Returns:
            Dictionary with processing statistics
        """
        # Load paper IDs
        paper_ids_path = Path(paper_ids_file)
        if not paper_ids_path.exists():
            raise FileNotFoundError(f"Paper IDs file not found: {paper_ids_file}")
        
        with open(paper_ids_path, 'r') as f:
            paper_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(paper_ids)} papers")
        
        return self.process_batch(paper_ids)
    
    def get_stats(self) -> Dict:
        """Get statistics about the processed data."""
        vector_stats = self.vector_store.get_stats() if hasattr(self, 'vector_store') else {}
        
        # Count processed papers
        processed_files = list(self.extracted_text_dir.glob("*.json"))
        
        # Get progress stats
        progress_stats = self.get_progress_stats()
        
        return {
            'vector_store': vector_stats,
            'processed_papers': len(processed_files),
            'extracted_text_dir': str(self.extracted_text_dir),
            'progress': progress_stats
        }

