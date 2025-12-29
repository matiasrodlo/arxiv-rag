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

from .pdf_extractor import PDFExtractor, load_metadata
from .text_processor import TextProcessor, TextChunker
from .embedder import Embedder
from .vector_store import VectorStore


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
            
            # Save extracted text for reference
            extracted_text_path = self.extracted_text_dir / f"{paper_id}.json"
            with open(extracted_text_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'paper_id': paper_id,
                    'metadata': metadata,
                    'text': cleaned_text,
                    'num_chunks': len(chunks),
                    'extraction_method': extraction_result['method_used']
                }, f, indent=2, ensure_ascii=False)
            
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

