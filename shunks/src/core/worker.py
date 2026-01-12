"""
Worker functions for parallel paper processing.
These functions are designed to be pickled and run in separate processes.
"""

import json
import sqlite3
import os
import time
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from ..extractors.pdf_extractor import PDFExtractor, load_metadata
from ..processors.text_processor import TextProcessor, TextChunker


def _process_paper_worker(pdf_dir: str, metadata_dir: str, extracted_text_dir: str,
                         progress_db_path: str, paper_id: str, config: Dict) -> Optional[Dict]:
    """
    Worker function to process a single paper.
    This function is designed to run in a separate process.
    
    Args:
        pdf_dir: Directory containing PDF files
        metadata_dir: Directory containing metadata files
        extracted_text_dir: Directory for extracted JSON files
        progress_db_path: Path to progress database
        paper_id: ArXiv paper ID
        config: Configuration dictionary
        
    Returns:
        Dictionary with processing results or None if failed
    """
    # #region agent log - VERY FIRST LOG
    log_path = Path("/Volumes/8SSD/DOIS/.cursor/debug.log")
    try:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            import json as json_lib
            log_entry = {
                "sessionId": "perf-test",
                "runId": "run1",
                "hypothesisId": "WORKER_ENTRY",
                "location": "worker.py:42",
                "message": "Worker function entered",
                "data": {
                    "paper_id": paper_id,
                    "pdf_dir": pdf_dir,
                    "pid": os.getpid()
                },
                "timestamp": int(time.time() * 1000)
            }
            log_file.write(json_lib.dumps(log_entry) + "\n")
            log_file.flush()  # Force write immediately
    except Exception as e:
        # If logging fails, we're in deep trouble - try to write to stderr
        import sys
        sys.stderr.write(f"CRITICAL: Worker logging failed for {paper_id}: {e}\n")
        sys.stderr.flush()
    # #endregion
    
    paper_start_time = time.time()
    
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            import json as json_lib
            if psutil_available:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                memory_mb = mem_info.rss / 1024 / 1024
            else:
                memory_mb = 0
            log_entry = {
                "sessionId": "perf-test",
                "runId": "run1",
                "hypothesisId": "PAPER_START",
                "location": "worker.py:36",
                "message": "Paper processing started",
                "data": {
                    "paper_id": paper_id,
                    "worker_pid": os.getpid(),
                    "memory_mb": memory_mb
                },
                "timestamp": int(time.time() * 1000)
            }
            log_file.write(json_lib.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
    
    try:
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "AFTER_PAPER_START",
                    "location": "worker.py:73",
                    "message": "Entered try block, checking paths",
                    "data": {
                        "paper_id": paper_id,
                        "extracted_text_dir": extracted_text_dir,
                        "cwd": os.getcwd(),
                        "resolved_path": str(Path(extracted_text_dir).resolve())
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Skip macOS resource fork files
        if paper_id.startswith('._'):
            return {
                'paper_id': paper_id,
                'success': False,
                'error': 'macOS resource fork file (not a real PDF)'
            }
        
        # Search for PDF in subdirectories (handles category/year_month structure)
        pdf_dir_path = Path(pdf_dir)
        pdf_path = pdf_dir_path / f"{paper_id}.pdf"
        if not pdf_path.exists():
            # Search recursively for the PDF
            found_pdfs = list(pdf_dir_path.rglob(f"{paper_id}.pdf"))
            if found_pdfs:
                pdf_path = found_pdfs[0]
        
        # Determine output path structure based on PDF location
        # Maintain same structure: output/{category}/{year_month}/{paper_id}.json
        if pdf_path.exists() and pdf_path.is_relative_to(pdf_dir_path):
            # Get relative path from pdf_dir (e.g., cs.AI/0709/0709.3974.pdf)
            relative_pdf_path = pdf_path.relative_to(pdf_dir_path)
            # Get parent directory structure (e.g., cs.AI/0709)
            output_subdir = relative_pdf_path.parent
            # Create output path maintaining structure
            extracted_text_path = Path(extracted_text_dir) / output_subdir / f"{paper_id}.json"
        else:
            # Fallback: flat structure if PDF not found or path issues
            extracted_text_path = Path(extracted_text_dir) / f"{paper_id}.json"
        
        metadata_path = Path(metadata_dir) / f"{paper_id}.txt"
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "BEFORE_PDF_CHECK",
                    "location": "worker.py:106",
                    "message": "About to check PDF existence",
                    "data": {
                        "paper_id": paper_id,
                        "pdf_path": str(pdf_path),
                        "pdf_dir": pdf_dir
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if not pdf_path.exists():
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "perf-test",
                        "runId": "run1",
                        "hypothesisId": "PDF_NOT_FOUND",
                        "location": "worker.py:110",
                        "message": "PDF file not found",
                        "data": {"paper_id": paper_id, "pdf_path": str(pdf_path)},
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            return {
                'paper_id': paper_id,
                'success': False,
                'error': f'PDF not found: {pdf_path}'
            }
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "AFTER_PDF_CHECK",
                    "location": "worker.py:135",
                    "message": "PDF exists, proceeding to initialization",
                    "data": {
                        "paper_id": paper_id,
                        "pdf_path": str(pdf_path),
                        "pdf_size_mb": pdf_path.stat().st_size / 1024 / 1024 if pdf_path.exists() else 0
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Initialize components (each worker needs its own instances)
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "INIT_START",
                    "location": "worker.py:117",
                    "message": "Starting component initialization",
                    "data": {"paper_id": paper_id},
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # #region agent log
        init_pdf_start = time.time()
        # #endregion
        # Maximum quality extraction settings
        pdf_extractor = PDFExtractor(
            enable_ocr=config['pdf_extraction']['enable_ocr'],
            ocr_language=config['pdf_extraction']['ocr_language'],
            max_retries=5,  # Increased for maximum quality
            enable_parallel=True,  # Enable parallel extraction
            max_workers=8,  # More workers for parallel
            large_pdf_threshold_mb=10.0,  # Lower threshold for parallel processing
            enable_caching=True  # Enable caching for performance
        )
        # #region agent log
        init_pdf_duration = time.time() - init_pdf_start
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "INIT_PDF",
                    "location": "worker.py:125",
                    "message": "PDF extractor initialized",
                    "data": {"paper_id": paper_id, "duration_seconds": init_pdf_duration},
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # #region agent log
        init_text_start = time.time()
        # #endregion
        text_processor = TextProcessor(
            remove_headers_footers=config['text_processing']['remove_headers_footers'],
            normalize_whitespace=config['text_processing']['normalize_whitespace'],
            fix_encoding=config['text_processing']['fix_encoding'],
            improve_formulas=config['text_processing'].get('improve_formulas', True)
        )
        # #region agent log
        init_text_duration = time.time() - init_text_start
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "INIT_TEXT",
                    "location": "worker.py:140",
                    "message": "Text processor initialized",
                    "data": {"paper_id": paper_id, "duration_seconds": init_text_duration},
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # #region agent log
        init_chunker_start = time.time()
        # #endregion
        chunker = TextChunker(
            method=config['chunking']['method'],
            chunk_size=config['chunking']['chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            model_name=config['chunking'].get('model'),
            min_chunk_size=config['text_processing']['min_chunk_size'],
            max_chunk_size=config['text_processing']['max_chunk_size'],
            device=config['chunking'].get('device', 'cpu'),
            batch_size=config['chunking'].get('batch_size', 512),
            enable_mixed_precision=config['chunking'].get('enable_mixed_precision', True)
        )
        # #region agent log
        init_chunker_duration = time.time() - init_chunker_start
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                # Get actual device used (chunker may have changed it from mps to cpu)
                actual_chunker_device = chunker.device if hasattr(chunker, 'device') else config['chunking'].get('device', 'cpu')
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "INIT_CHUNKER",
                    "location": "worker.py:260",
                    "message": "Text chunker initialized",
                    "data": {
                        "paper_id": paper_id,
                        "duration_seconds": init_chunker_duration,
                        "requested_device": config['chunking'].get('device', 'cpu'),
                        "actual_device": actual_chunker_device,
                        "model": config['chunking'].get('model', 'unknown')
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Step 1: Extract PDF text
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "BEFORE_EXTRACTION",
                    "location": "worker.py:200",
                    "message": "About to start PDF extraction",
                    "data": {
                        "paper_id": paper_id,
                        "pdf_path": str(pdf_path),
                        "pdf_exists": pdf_path.exists(),
                        "pdf_size_mb": pdf_path.stat().st_size / 1024 / 1024 if pdf_path.exists() else 0
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        extraction_start = time.time()
        # #endregion
        extraction_result = pdf_extractor.extract(str(pdf_path))
        # #region agent log
        extraction_duration = time.time() - extraction_start
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "STAGE_EXTRACTION",
                    "location": "worker.py:106",
                    "message": "PDF extraction completed",
                    "data": {
                        "paper_id": paper_id,
                        "duration_seconds": extraction_duration,
                        "success": extraction_result.get('success', False),
                        "method_used": extraction_result.get('method_used', 'unknown')
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if not extraction_result['success']:
            return {
                'paper_id': paper_id,
                'success': False,
                'error': extraction_result.get('error', 'Extraction failed')
            }
        
        # Step 2: Load metadata
        metadata = load_metadata(str(metadata_path)) if metadata_path.exists() else {}
        metadata['paper_id'] = paper_id
        metadata['extraction_method'] = extraction_result['method_used']
        
        # Step 3: Clean text
        # #region agent log
        cleaning_start = time.time()
        # #endregion
        cleaned_text = text_processor.clean(extraction_result['text'])
        # #region agent log
        cleaning_duration = time.time() - cleaning_start
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "STAGE_CLEANING",
                    "location": "worker.py:121",
                    "message": "Text cleaning completed",
                    "data": {
                        "paper_id": paper_id,
                        "duration_seconds": cleaning_duration,
                        "text_length": len(cleaned_text)
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if len(cleaned_text) < config['pdf_extraction']['min_text_length']:
            return {
                'paper_id': paper_id,
                'success': False,
                'error': f'Text too short: {len(cleaned_text)} chars'
            }
        
        # Step 4: Chunk text
        # #region agent log
        chunking_start = time.time()
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "BEFORE_CHUNKING",
                    "location": "worker.py:391",
                    "message": "About to call chunker.chunk",
                    "data": {
                        "paper_id": paper_id,
                        "text_length": len(cleaned_text),
                        "chunker_method": chunker.method,
                        "chunker_device": chunker.device
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        try:
            chunks = chunker.chunk(cleaned_text, metadata=metadata)
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "perf-test",
                        "runId": "run1",
                        "hypothesisId": "AFTER_CHUNKER_CALL",
                        "location": "worker.py:413",
                        "message": "chunker.chunk returned successfully",
                        "data": {
                            "paper_id": paper_id,
                            "num_chunks": len(chunks) if chunks else 0
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
        except Exception as chunk_error:
            # #region agent log
            chunking_duration = time.time() - chunking_start
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "perf-test",
                        "runId": "run1",
                        "hypothesisId": "CHUNKING_ERROR",
                        "location": "worker.py:391",
                        "message": "Chunking failed with exception",
                        "data": {
                            "paper_id": paper_id,
                            "duration_seconds": chunking_duration,
                            "error": str(chunk_error),
                            "error_type": type(chunk_error).__name__
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            raise
        
        # #region agent log
        chunking_duration = time.time() - chunking_start
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "STAGE_CHUNKING",
                    "location": "worker.py:420",
                    "message": "Text chunking completed",
                    "data": {
                        "paper_id": paper_id,
                        "duration_seconds": chunking_duration,
                        "num_chunks": len(chunks) if chunks else 0,
                        "device": chunker.device,
                        "method": chunker.method,
                        "batch_size": config['chunking'].get('batch_size', 512)
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if not chunks:
            return {
                'paper_id': paper_id,
                'success': False,
                'error': 'No chunks created'
            }
        
        # Step 5: Extract sections
        pages_data = extraction_result.get('pages', [])
        sections = text_processor.extract_sections(cleaned_text, pages=pages_data)
        
        # Step 8: Extract citations and enhanced metadata
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
        
        # Step 9: Prepare enhanced JSON structure with section/page metadata
        # Pre-build section boundaries for faster lookup
        section_boundaries = []
        for s in sections:
            start_char = s.get('start_char', 0)
            end_char = s.get('end_char', start_char + len(s.get('text', '')))
            section_boundaries.append((start_char, end_char, s.get('name', 'Unknown')))
        section_boundaries.sort()  # Sort by start_char
        
        def find_section_for_position(pos: int) -> str:
            """Find which section a character position belongs to (binary search)."""
            if not section_boundaries:
                return 'Unknown'
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
            # If position is beyond all sections, return last section
            if section_boundaries and pos >= section_boundaries[-1][1]:
                return section_boundaries[-1][2]
            return 'Unknown'
        
        # Pre-build page boundaries for faster lookup (based on actual page text positions)
        page_boundaries = []
        current_pos = 0
        for page in pages_data:
            page_text = page.get('text', '')
            page_length = len(page_text)
            page_num = page.get('page', len(page_boundaries) + 1)
            page_boundaries.append((current_pos, current_pos + page_length, page_num))
            current_pos += page_length + 2  # +2 for page separator (\n\n)
        
        def find_page_for_position(pos: int) -> int:
            """Find which page a character position belongs to (binary search)."""
            if not page_boundaries:
                return 1
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
            # If position is beyond all pages, return last page
            if page_boundaries and pos >= page_boundaries[-1][1]:
                return page_boundaries[-1][2]
            return page_boundaries[-1][2] if page_boundaries else 1
        
        # Build JSON structure
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
                            # Use actual character positions from chunk metadata for accurate section/page mapping
                            'section': find_section_for_position(chunk.get('metadata', {}).get('char_start', 0)),
                            'page': find_page_for_position(chunk.get('metadata', {}).get('char_start', 0)),
                            # Also include section/page for the end of chunk (in case chunk spans multiple sections/pages)
                            'section_end': find_section_for_position(chunk.get('metadata', {}).get('char_end', len(cleaned_text))),
                            'page_end': find_page_for_position(chunk.get('metadata', {}).get('char_end', len(cleaned_text)))
                    }
                }
                    for i, chunk in enumerate(chunks)
            ],
            'statistics': {
                'num_chunks': len(chunks),
                'num_pages': len(pages_data),
                'num_sections': len(sections),
                'total_chars': len(cleaned_text),
                'avg_chunk_size': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0,
                'chunking_method': chunker.method,
                'total_citations': citations_data.get('total_citations', 0),
                'total_references': citations_data.get('total_references', 0)
            }
        }
        
        # Save JSON file (formatted for readability)
        # #region agent log
        log_path = Path("/Volumes/8SSD/DOIS/.cursor/debug.log")
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "worker.py:243",
                    "message": "Before saving JSON file",
                    "data": {
                        "paper_id": paper_id,
                        "extracted_text_path": str(extracted_text_path),
                        "absolute_path": str(extracted_text_path.resolve()),
                        "parent_exists": extracted_text_path.parent.exists(),
                        "file_will_exist": extracted_text_path.exists()
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Ensure directory exists
        try:
            extracted_text_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as dir_error:
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "worker.py:294",
                        "message": "Directory creation failed",
                        "data": {
                            "paper_id": paper_id,
                            "path": str(extracted_text_path.parent),
                            "error": str(dir_error)
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            raise
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "F",
                    "location": "worker.py:310",
                    "message": "About to write file",
                    "data": {
                        "paper_id": paper_id,
                        "extracted_text_path": str(extracted_text_path),
                        "absolute_path": str(extracted_text_path.resolve()),
                        "parent_exists": extracted_text_path.parent.exists(),
                        "parent_writable": os.access(extracted_text_path.parent, os.W_OK) if extracted_text_path.parent.exists() else False
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Sanitize text data to remove surrogate characters that can't be encoded in UTF-8
        def sanitize_text(obj):
            """Recursively sanitize text in data structures to remove surrogate characters."""
            if isinstance(obj, str):
                # Replace surrogate characters with replacement character
                return obj.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            elif isinstance(obj, dict):
                return {k: sanitize_text(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_text(item) for item in obj]
            else:
                return obj
        
        sanitized_data = sanitize_text(extracted_data)
        
        try:
            with open(extracted_text_path, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(sanitized_data, f, indent=2, ensure_ascii=False)
        except Exception as write_error:
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "G",
                        "location": "worker.py:330",
                        "message": "File write failed",
                        "data": {
                            "paper_id": paper_id,
                            "extracted_text_path": str(extracted_text_path),
                            "error": str(write_error),
                            "error_type": type(write_error).__name__
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            raise
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "worker.py:260",
                    "message": "After saving JSON file",
                    "data": {
                        "paper_id": paper_id,
                        "extracted_text_path": str(extracted_text_path),
                        "absolute_path": str(extracted_text_path.resolve()),
                        "file_exists": extracted_text_path.exists(),
                        "file_size": extracted_text_path.stat().st_size if extracted_text_path.exists() else 0
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # #region agent log
        paper_end_time = time.time()
        total_duration = paper_end_time - paper_start_time
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                if psutil_available:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    memory_mb = mem_info.rss / 1024 / 1024
                else:
                    memory_mb = 0
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "PAPER_END",
                    "location": "worker.py:427",
                    "message": "Paper processing completed",
                    "data": {
                        "paper_id": paper_id,
                        "total_duration_seconds": total_duration,
                        "num_chunks": len(chunks),
                        "text_length": len(cleaned_text),
                        "memory_mb": memory_mb,
                        "success": True
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        return {
            'paper_id': paper_id,
            'success': True,
            'num_chunks': len(chunks),
            'text_length': len(cleaned_text),
            'extraction_method': extraction_result['method_used']
        }
        
    except Exception as e:
        # #region agent log
        paper_end_time = time.time()
        total_duration = paper_end_time - paper_start_time
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "perf-test",
                    "runId": "run1",
                    "hypothesisId": "PAPER_ERROR",
                    "location": "worker.py:435",
                    "message": "Paper processing failed",
                    "data": {
                        "paper_id": paper_id,
                        "total_duration_seconds": total_duration,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "success": False
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        return {
            'paper_id': paper_id,
            'success': False,
            'error': str(e)
        }

