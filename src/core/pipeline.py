"""
Main RAG Pipeline
Orchestrates the complete pipeline from PDF extraction to vector store indexing.
"""

import os
import json
import sqlite3
import time
import psutil
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

# Import memory optimizer
try:
    from .memory_optimizer import MemoryOptimizer
except ImportError:
    MemoryOptimizer = None

# Import worker function for multiprocessing
try:
    from .worker import _process_paper_worker
except ImportError:
    # Fallback: define worker function here if import fails
    _process_paper_worker = None


class RAGPipeline:
    """Complete RAG pipeline for processing ArXiv papers."""
    
    def __init__(self, config_path: str = "config.yaml", enable_memory_optimization: bool = True):
        """Initialize pipeline with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup memory optimization if enabled
        self.memory_optimizer = None
        if enable_memory_optimization and MemoryOptimizer:
            try:
                self.memory_optimizer = MemoryOptimizer(
                    use_ram_disk=self.config.get('memory_optimization', {}).get('use_ram_disk', True),
                    ram_disk_size_gb=self.config.get('memory_optimization', {}).get('ram_disk_size_gb', 20),
                    enable_model_caching=self.config.get('memory_optimization', {}).get('enable_model_caching', True),
                    max_workers=self.config.get('processing', {}).get('num_workers', 12)
                )
                
                # Setup RAM disk for cache
                ram_disk_path = self.memory_optimizer.setup_ram_disk()
                if ram_disk_path:
                    logger.info(f"Memory optimization enabled. RAM disk: {ram_disk_path}")
                
                # Optimize configuration
                self.config = self.memory_optimizer.optimize_worker_memory(self.config)
                
                # Log memory stats
                mem_stats = self.memory_optimizer.get_memory_stats()
                if mem_stats:
                    logger.info(f"Memory stats: {mem_stats.get('system_available_gb', 0):.1f}GB available "
                              f"({mem_stats.get('system_free_percent', 0):.1f}% free)")
                    
                    # Recommend worker count
                    recommended = self.memory_optimizer.recommend_worker_count()
                    current = self.config.get('processing', {}).get('num_workers', 12)
                    if recommended != current:
                        logger.info(f"💡 Memory-based recommendation: {recommended} workers (current: {current})")
                        logger.info(f"   Consider updating num_workers in config.yaml if you want to use more RAM")
            except Exception as e:
                logger.warning(f"Memory optimization setup failed: {e}. Continuing without optimization.")
                self.memory_optimizer = None
        
        # Initialize components with memory-optimized cache if available
        cache_dir = None
        if self.memory_optimizer:
            base_cache = Path.home() / '.arxiv_rag_cache'
            cache_dir = str(self.memory_optimizer.get_cache_directory(base_cache))
        
        self.pdf_extractor = PDFExtractor(
            enable_ocr=self.config['pdf_extraction']['enable_ocr'],
            ocr_language=self.config['pdf_extraction']['ocr_language'],
            enable_caching=self.config.get('pdf_extraction', {}).get('enable_caching', True),
            cache_dir=cache_dir
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
            max_chunk_size=self.config['text_processing']['max_chunk_size'],
            device=self.config['chunking'].get('device', 'cpu'),
            batch_size=self.config['chunking'].get('batch_size', 512),
            enable_mixed_precision=self.config['chunking'].get('enable_mixed_precision', True)
        )
        
        # Initialize embeddings only if needed during processing
        self.embedder = None
        if self.config.get('embeddings', {}).get('generate_during_processing', True):
            try:
                self.embedder = Embedder(
                    model_name=self.config['embeddings']['model'],
                    batch_size=self.config['embeddings']['batch_size'],
                    device=self.config['embeddings']['device'],
                    normalize_embeddings=self.config['embeddings']['normalize_embeddings'],
                    enable_mixed_precision=self.config['embeddings'].get('enable_mixed_precision', True),
                    enable_pipelining=self.config['embeddings'].get('enable_pipelining', False)
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
        
        # Initialize advanced optimizer if enabled
        self.advanced_optimizer = None
        if self.config.get('advanced_optimization', {}).get('enable_preloading', False) or \
           self.config.get('advanced_optimization', {}).get('enable_async_io', False):
            try:
                from .advanced_optimizations import create_advanced_optimizer
                self.advanced_optimizer = create_advanced_optimizer(self.config)
                if self.advanced_optimizer:
                    self.advanced_optimizer.setup_pdf_cache(self.pdf_dir)
                    logger.info("Advanced optimizer initialized (PDF cache + Async I/O)")
            except Exception as e:
                logger.warning(f"Advanced optimizer setup failed: {e}. Continuing without it.")
        
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
            
            # Extract citations and enhanced metadata
            try:
                citations_data = TextProcessor.extract_citations(cleaned_text, sections=sections)
            except Exception as e:
                logger.warning(f"Citation extraction failed for {paper_id}: {e}")
                citations_data = {'in_text': [], 'references': [], 'total_citations': 0, 'total_references': 0}
            
            try:
                enhanced_metadata = TextProcessor.extract_metadata(cleaned_text, sections=sections)
                # Merge enhanced metadata into existing metadata
                metadata.update(enhanced_metadata)
            except Exception as e:
                logger.warning(f"Metadata extraction failed for {paper_id}: {e}")
                enhanced_metadata = {}
            
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
                'citations': citations_data,
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
                    'chunking_method': self.chunker.method,
                    'total_citations': citations_data.get('total_citations', 0),
                    'total_references': citations_data.get('total_references', 0)
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
        # #region agent log
        log_path = Path("/Volumes/8SSD/ArxivCS/arxiv-rag/.cursor/debug.log")
        original_count = len(paper_ids)  # Store original count at function start
        skipped = 0  # Initialize skipped count
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "PROCESS_BATCH_ENTRY",
                    "location": "pipeline.py:500",
                    "message": "process_batch method entered",
                    "data": {
                        "num_papers": len(paper_ids),
                        "batch_size": batch_size,
                        "skip_processed": skip_processed,
                        "num_workers": num_workers
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
                log_file.flush()
        except Exception:
            pass
        # #endregion
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
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "BEFORE_CONFIG_READ",
                    "location": "pipeline.py:514",
                    "message": "About to read config for batch_size and num_workers",
                    "data": {},
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
                log_file.flush()
        except Exception:
            pass
        # #endregion
        
        batch_size = batch_size or self.config['processing']['batch_size']
        num_workers = num_workers or self.config['processing']['num_workers']
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "AFTER_CONFIG_READ",
                    "location": "pipeline.py:530",
                    "message": "Config read, about to filter paper_ids",
                    "data": {
                        "batch_size": batch_size,
                        "num_workers": num_workers
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
                log_file.flush()
        except Exception:
            pass
        # #endregion
        
        # Filter out macOS resource fork files (._*)
        original_count = len(paper_ids)
        paper_ids = [pid for pid in paper_ids if not pid.startswith('._')]
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "AFTER_FILTER",
                    "location": "pipeline.py:550",
                    "message": "Paper IDs filtered",
                    "data": {
                        "original_count": original_count,
                        "filtered_count": len(paper_ids)
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
                log_file.flush()
        except Exception:
            pass
        # #endregion
        macos_filtered = original_count - len(paper_ids)
        if macos_filtered > 0:
            logger.info(f"Filtered out {macos_filtered} macOS resource fork files")
        
        # Filter out already processed papers
        skipped = 0
        if skip_processed:
            original_count = len(paper_ids)
            logger.info(f"Checking {original_count} papers for already processed status...")
            # #region agent log
            log_path = Path("/Volumes/8SSD/ArxivCS/arxiv-rag/.cursor/debug.log")
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
            
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "perf-test",
                        "runId": "run1",
                        "hypothesisId": "BEFORE_DB_CHECK",
                        "location": "pipeline.py:576",
                        "message": "About to check database for processed papers",
                        "data": {
                            "num_papers": len(paper_ids),
                            "db_path": str(self.progress_db_path)
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
                    log_file.flush()
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
            
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "perf-test",
                        "runId": "run1",
                        "hypothesisId": "AFTER_DB_CHECK",
                        "location": "pipeline.py:590",
                        "message": "Database check completed",
                        "data": {
                            "processed_count": len(processed_set)
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
                    log_file.flush()
            except Exception:
                pass
            # #endregion
            
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
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "BEFORE_PAPER_IDS_CHECK",
                    "location": "pipeline.py:720",
                    "message": "About to check if paper_ids is empty",
                    "data": {
                        "paper_ids_count": len(paper_ids),
                        "skipped": skipped
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
                log_file.flush()
        except Exception:
            pass
        # #endregion
        
        if not paper_ids:
            logger.info("All papers already processed")
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "perf-test",
                        "runId": "run1",
                        "hypothesisId": "ALL_PAPERS_SKIPPED",
                        "location": "pipeline.py:615",
                        "message": "All papers already processed, returning early",
                        "data": {
                            "total": original_count if skip_processed else 0,
                            "skipped": skipped
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
                    log_file.flush()
            except Exception:
                pass
            # #endregion
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
        
        # Preload first batch of PDFs if advanced optimizer is enabled
        if self.advanced_optimizer and self.advanced_optimizer.enable_preloading:
            first_batch = paper_ids[:min(batch_size, len(paper_ids))]
            logger.info(f"Preloading {len(first_batch)} PDFs into memory cache...")
            self.advanced_optimizer.preload_pdfs(first_batch, background=True)
        
        # Process in batches with parallel workers
        total_batches = (len(paper_ids) + batch_size - 1) // batch_size
        
        # #region agent log
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
        overall_start_time = time.time()
        log_path = Path("/Volumes/8SSD/ArxivCS/arxiv-rag/.cursor/debug.log")
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                if psutil_available:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    initial_memory_mb = mem_info.rss / 1024 / 1024
                    cpu_count = psutil.cpu_count()
                else:
                    initial_memory_mb = 0
                    cpu_count = os.cpu_count() or 0
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "PERF_START",
                    "location": "pipeline.py:613",
                    "message": "Batch processing started",
                    "data": {
                        "total_papers": len(paper_ids),
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "total_batches": total_batches,
                        "initial_memory_mb": initial_memory_mb,
                        "cpu_count": cpu_count
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        with tqdm(total=len(paper_ids), desc="Processing papers") as pbar:
            for batch_idx in range(0, len(paper_ids), batch_size):
                batch = paper_ids[batch_idx:batch_idx + batch_size]
                
                # #region agent log
                batch_start_time = time.time()
                try:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        import json as json_lib
                        if psutil_available:
                            process = psutil.Process(os.getpid())
                            mem_info = process.memory_info()
                            memory_mb = mem_info.rss / 1024 / 1024
                            cpu_percent = psutil.cpu_percent(interval=0.1)
                        else:
                            memory_mb = 0
                            cpu_percent = 0
                        log_entry = {
                            "sessionId": "perf-test",
                            "runId": "run1",
                            "hypothesisId": "BATCH_START",
                            "location": "pipeline.py:617",
                            "message": "Batch processing started",
                            "data": {
                                "batch_idx": batch_idx // batch_size + 1,
                                "batch_size": len(batch),
                                "papers_processed_so_far": batch_idx,
                                "memory_mb": memory_mb,
                                "cpu_percent": cpu_percent
                            },
                            "timestamp": int(time.time() * 1000)
                        }
                        log_file.write(json_lib.dumps(log_entry) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # Preload next batch while processing current batch
                if self.advanced_optimizer and self.advanced_optimizer.enable_preloading:
                    self.advanced_optimizer.preload_next_batch(batch, paper_ids)
                
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
                    
                    # #region agent log
                    try:
                        with open(log_path, 'a', encoding='utf-8') as log_file:
                            import json as json_lib
                            log_entry = {
                                "sessionId": "perf-test",
                                "runId": "run1",
                                "hypothesisId": "BEFORE_EXECUTOR",
                                "location": "pipeline.py:741",
                                "message": "About to create ProcessPoolExecutor",
                                "data": {
                                    "batch_idx": batch_idx // batch_size + 1,
                                    "batch_size": len(batch),
                                    "num_workers": num_workers
                                },
                                "timestamp": int(time.time() * 1000)
                            }
                            log_file.write(json_lib.dumps(log_entry) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        # #region agent log
                        try:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                import json as json_lib
                                log_entry = {
                                    "sessionId": "perf-test",
                                    "runId": "run1",
                                    "hypothesisId": "EXECUTOR_CREATED",
                                    "location": "pipeline.py:760",
                                    "message": "ProcessPoolExecutor created",
                                    "data": {
                                        "batch_idx": batch_idx // batch_size + 1,
                                        "num_workers": num_workers
                                    },
                                    "timestamp": int(time.time() * 1000)
                                }
                                log_file.write(json_lib.dumps(log_entry) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        # Submit all papers in batch
                        # #region agent log
                        try:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                import json as json_lib
                                log_entry = {
                                    "sessionId": "perf-test",
                                    "runId": "run1",
                                    "hypothesisId": "BEFORE_SUBMIT",
                                    "location": "pipeline.py:790",
                                    "message": "About to submit workers",
                                    "data": {
                                        "batch_idx": batch_idx // batch_size + 1,
                                        "batch_size": len(batch),
                                        "num_papers": len(batch)
                                    },
                                    "timestamp": int(time.time() * 1000)
                                }
                                log_file.write(json_lib.dumps(log_entry) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # #region agent log
                        try:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                import json as json_lib
                                log_entry = {
                                    "sessionId": "perf-test",
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
                        
                        # #region agent log
                        try:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                import json as json_lib
                                log_entry = {
                                    "sessionId": "perf-test",
                                    "runId": "run1",
                                    "hypothesisId": "SUBMITTING_WORKERS",
                                    "location": "pipeline.py:800",
                                    "message": "Submitting workers to executor",
                                    "data": {
                                        "batch_idx": batch_idx // batch_size + 1,
                                        "num_papers": len(batch),
                                        "paper_ids": list(batch)[:5]  # First 5 for logging
                                    },
                                    "timestamp": int(time.time() * 1000)
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
                        
                        # #region agent log
                        try:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                import json as json_lib
                                log_entry = {
                                    "sessionId": "perf-test",
                                    "runId": "run1",
                                    "hypothesisId": "WORKERS_SUBMITTED",
                                    "location": "pipeline.py:815",
                                    "message": "All workers submitted to executor",
                                    "data": {
                                        "batch_idx": batch_idx // batch_size + 1,
                                        "num_futures": len(future_to_paper)
                                    },
                                    "timestamp": int(time.time() * 1000)
                                }
                                log_file.write(json_lib.dumps(log_entry) + "\n")
                        except Exception:
                            pass
                        # #endregion
                    
                        # Collect results as they complete
                        for future in as_completed(future_to_paper):
                            paper_id = future_to_paper[future]
                            # #region agent log
                            try:
                                with open(log_path, 'a', encoding='utf-8') as log_file:
                                    import json as json_lib
                                    log_entry = {
                                        "sessionId": "perf-test",
                                        "runId": "run1",
                                        "hypothesisId": "BEFORE_FUTURE_RESULT",
                                        "location": "pipeline.py:881",
                                        "message": "About to get future.result()",
                                        "data": {
                                            "paper_id": paper_id,
                                            "future_done": future.done(),
                                            "future_cancelled": future.cancelled()
                                        },
                                        "timestamp": int(time.time() * 1000)
                                    }
                                    log_file.write(json_lib.dumps(log_entry) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            
                            try:
                                result = future.result()
                                # #region agent log
                                try:
                                    with open(log_path, 'a', encoding='utf-8') as log_file:
                                        import json as json_lib
                                        log_entry = {
                                            "sessionId": "perf-test",
                                            "runId": "run1",
                                            "hypothesisId": "FUTURE_RESULT_RECEIVED",
                                            "location": "pipeline.py:900",
                                            "message": "future.result() returned",
                                            "data": {
                                                "paper_id": paper_id,
                                                "result_success": result.get('success') if result else None,
                                                "result_error": result.get('error') if result else None,
                                                "result_is_none": result is None
                                            },
                                            "timestamp": int(time.time() * 1000)
                                        }
                                        log_file.write(json_lib.dumps(log_entry) + "\n")
                                except Exception:
                                    pass
                                # #endregion
                                
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
                                # #region agent log
                                try:
                                    with open(log_path, 'a', encoding='utf-8') as log_file:
                                        import json as json_lib
                                        log_entry = {
                                            "sessionId": "perf-test",
                                            "runId": "run1",
                                            "hypothesisId": "FUTURE_RESULT_EXCEPTION",
                                            "location": "pipeline.py:930",
                                            "message": "Exception getting future.result()",
                                            "data": {
                                                "paper_id": paper_id,
                                                "error": str(e),
                                                "error_type": type(e).__name__
                                            },
                                            "timestamp": int(time.time() * 1000)
                                        }
                                        log_file.write(json_lib.dumps(log_entry) + "\n")
                                except Exception:
                                    pass
                                # #endregion
                                
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
                
                # #region agent log
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                try:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        import json as json_lib
                        if psutil_available:
                            process = psutil.Process(os.getpid())
                            mem_info = process.memory_info()
                            memory_mb = mem_info.rss / 1024 / 1024
                            cpu_percent = psutil.cpu_percent(interval=0.1)
                        else:
                            memory_mb = 0
                            cpu_percent = 0
                        log_entry = {
                            "sessionId": "perf-test",
                            "runId": "run1",
                            "hypothesisId": "BATCH_END",
                            "location": "pipeline.py:742",
                            "message": "Batch processing completed",
                            "data": {
                                "batch_idx": batch_idx // batch_size + 1,
                                "batch_size": len(batch),
                                "duration_seconds": batch_duration,
                                "papers_per_second": len(batch) / batch_duration if batch_duration > 0 else 0,
                                "memory_mb": memory_mb,
                                "cpu_percent": cpu_percent
                            },
                            "timestamp": int(time.time() * 1000)
                        }
                        log_file.write(json_lib.dumps(log_entry) + "\n")
                except Exception:
                    pass
                # #endregion
        
        # #region agent log
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                if psutil_available:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    final_memory_mb = mem_info.rss / 1024 / 1024
                else:
                    final_memory_mb = 0
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "PERF_END",
                    "location": "pipeline.py:744",
                    "message": "Overall batch processing completed",
                    "data": {
                        "total_papers": len(paper_ids),
                        "successful": results['successful'],
                        "failed": results['failed'],
                        "total_duration_seconds": overall_duration,
                        "papers_per_second": results['successful'] / overall_duration if overall_duration > 0 else 0,
                        "papers_per_hour": (results['successful'] / overall_duration * 3600) if overall_duration > 0 else 0,
                        "final_memory_mb": final_memory_mb
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
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
        vector_stats = self.vector_store.get_stats() if (hasattr(self, 'vector_store') and self.vector_store is not None) else {}
        
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

