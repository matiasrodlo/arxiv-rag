"""
Worker functions for parallel paper processing.
These functions are designed to be pickled and run in separate processes.
"""

import json
import sqlite3
import os
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from ..extractors.pdf_extractor import PDFExtractor, load_metadata
from ..processors.text_processor import TextProcessor, TextChunker
from ..embeddings.embedder import Embedder
from ..storage.vector_store import VectorStore


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
    try:
        # #region agent log
        log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "worker.py:36",
                    "message": "Worker entry - path resolution",
                    "data": {
                        "paper_id": paper_id,
                        "extracted_text_dir_param": extracted_text_dir,
                        "cwd": os.getcwd(),
                        "resolved_path": str(Path(extracted_text_dir).resolve())
                    },
                    "timestamp": int(__import__('time').time() * 1000)
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
        
        pdf_path = Path(pdf_dir) / f"{paper_id}.pdf"
        metadata_path = Path(metadata_dir) / f"{paper_id}.txt"
        extracted_text_path = Path(extracted_text_dir) / f"{paper_id}.json"
        
        if not pdf_path.exists():
            return {
                'paper_id': paper_id,
                'success': False,
                'error': f'PDF not found: {pdf_path}'
            }
        
        # Initialize components (each worker needs its own instances)
        pdf_extractor = PDFExtractor(
            enable_ocr=config['pdf_extraction']['enable_ocr'],
            ocr_language=config['pdf_extraction']['ocr_language']
        )
        
        text_processor = TextProcessor(
            remove_headers_footers=config['text_processing']['remove_headers_footers'],
            normalize_whitespace=config['text_processing']['normalize_whitespace'],
            fix_encoding=config['text_processing']['fix_encoding'],
            improve_formulas=config['text_processing'].get('improve_formulas', True)
        )
        
        chunker = TextChunker(
            method=config['chunking']['method'],
            chunk_size=config['chunking']['chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            model_name=config['chunking'].get('model'),
            min_chunk_size=config['text_processing']['min_chunk_size'],
            max_chunk_size=config['text_processing']['max_chunk_size']
        )
        
        # Step 1: Extract PDF text
        extraction_result = pdf_extractor.extract(str(pdf_path))
        
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
        cleaned_text = text_processor.clean(extraction_result['text'])
        
        if len(cleaned_text) < config['pdf_extraction']['min_text_length']:
            return {
                'paper_id': paper_id,
                'success': False,
                'error': f'Text too short: {len(cleaned_text)} chars'
            }
        
        # Step 4: Chunk text
        chunks = chunker.chunk(cleaned_text, metadata=metadata)
        
        if not chunks:
            return {
                'paper_id': paper_id,
                'success': False,
                'error': 'No chunks created'
            }
        
        # Step 5: Generate embeddings (only if embeddings are enabled)
        # For large-scale processing, we might skip embeddings here and do batch embedding later
        chunks_with_embeddings = chunks
        if config.get('embeddings', {}).get('generate_during_processing', True):
            try:
                embedder = Embedder(
                    model_name=config['embeddings']['model'],
                    batch_size=config['embeddings']['batch_size'],
                    device=config['embeddings']['device'],
                    normalize_embeddings=config['embeddings']['normalize_embeddings']
                )
                chunks_with_embeddings = embedder.embed_chunks(chunks, show_progress=False)
            except Exception as e:
                logger.warning(f"Embedding generation failed for {paper_id}: {e}, continuing without embeddings")
        
        # Step 6: Add to vector store (only if vector store is enabled)
        if config.get('vector_db', {}).get('add_during_processing', True):
            try:
                vector_store = VectorStore(
                    db_type=config['vector_db']['type'],
                    collection_name=config['vector_db']['collection_name'],
                    persist_directory=config['vector_db'].get('persist_directory'),
                    qdrant_host=config['vector_db'].get('qdrant_host', 'localhost'),
                    qdrant_port=config['vector_db'].get('qdrant_port', 6333)
                )
                vector_store.add_chunks(chunks_with_embeddings)
            except Exception as e:
                logger.warning(f"Vector store addition failed for {paper_id}: {e}, continuing")
        
        # Step 7: Extract sections
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
        
        # Step 9: Prepare enhanced JSON structure
        # Pre-build section boundaries for faster lookup
        section_boundaries = [(s['start_char'], s['end_char'], s['name']) for s in sections]
        section_boundaries.sort()
        
        def find_section_for_position(pos: int) -> str:
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
        
        # Pre-build page boundaries
        page_boundaries = []
        current_pos = 0
        for page in pages_data:
            page_text = page.get('text', '')
            page_length = len(page_text)
            page_boundaries.append((current_pos, current_pos + page_length, page.get('page', 1)))
            current_pos += page_length + 2
        
        def find_page_for_position(pos: int) -> int:
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
        
        # Prepare chunk start positions
        chunk_start_positions = []
        current_pos = 0
        for chunk in chunks:
            chunk_start_positions.append(current_pos)
            current_pos += len(chunk['text']) + 1
        
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
                'chunking_method': chunker.method,
                'total_citations': citations_data.get('total_citations', 0),
                'total_references': citations_data.get('total_references', 0)
            }
        }
        
        # Save JSON file (formatted for readability)
        # #region agent log
        log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
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
        
        try:
            with open(extracted_text_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
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
        
        return {
            'paper_id': paper_id,
            'success': True,
            'num_chunks': len(chunks),
            'text_length': len(cleaned_text),
            'extraction_method': extraction_result['method_used']
        }
        
    except Exception as e:
        return {
            'paper_id': paper_id,
            'success': False,
            'error': str(e)
        }

