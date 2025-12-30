"""
Main RAG Pipeline
Orchestrates the complete pipeline from PDF extraction to vector store indexing.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
import yaml

from ..extractors.pdf_extractor import PDFExtractor, load_metadata
from ..processors.text_processor import TextProcessor, TextChunker
from ..embeddings.embedder import Embedder
from ..storage.vector_store import VectorStore


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
        
        self.embedder = Embedder(
            model_name=self.config['embeddings']['model'],
            batch_size=self.config['embeddings']['batch_size'],
            device=self.config['embeddings']['device'],
            normalize_embeddings=self.config['embeddings']['normalize_embeddings']
        )
        
        self.vector_store = VectorStore(
            db_type=self.config['vector_db']['type'],
            collection_name=self.config['vector_db']['collection_name'],
            persist_directory=self.config['vector_db'].get('persist_directory'),
            qdrant_host=self.config['vector_db'].get('qdrant_host', 'localhost'),
            qdrant_port=self.config['vector_db'].get('qdrant_port', 6333)
        )
        
        # Setup paths
        self.pdf_dir = Path(self.config['paths']['pdf_dir'])
        self.metadata_dir = Path(self.config['paths']['metadata_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.extracted_text_dir = Path(self.config['paths']['extracted_text_dir'])
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RAG Pipeline initialized")
    
    def process_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Process a single paper through the complete pipeline.
        
        Args:
            paper_id: ArXiv paper ID (e.g., "2511.22108v1")
            
        Returns:
            Dictionary with processing results or None if failed
        """
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
            
            # Step 5: Generate embeddings
            chunks_with_embeddings = self.embedder.embed_chunks(chunks, show_progress=False)
            
            # Step 6: Add to vector store
            self.vector_store.add_chunks(chunks_with_embeddings)
            
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
            
            # Save enhanced JSON structure
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
    
    def process_batch(self, paper_ids: List[str], batch_size: Optional[int] = None) -> Dict:
        """
        Process a batch of papers.
        
        Args:
            paper_ids: List of paper IDs to process
            batch_size: Optional batch size (from config if not provided)
            
        Returns:
            Dictionary with processing statistics
        """
        batch_size = batch_size or self.config['processing']['batch_size']
        
        results = {
            'total': len(paper_ids),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        # Process in batches
        for i in tqdm(range(0, len(paper_ids), batch_size), desc="Processing batches"):
            batch = paper_ids[i:i + batch_size]
            
            for paper_id in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
                result = self.process_paper(paper_id)
                
                if result and result.get('success'):
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    if result:
                        results['errors'].append({
                            'paper_id': paper_id,
                            'error': result.get('error', 'Unknown error')
                        })
        
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
        vector_stats = self.vector_store.get_stats()
        
        # Count processed papers
        processed_files = list(self.extracted_text_dir.glob("*.json"))
        
        return {
            'vector_store': vector_stats,
            'processed_papers': len(processed_files),
            'extracted_text_dir': str(self.extracted_text_dir)
        }

