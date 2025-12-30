"""
Embedding Generation Module
Creates vector embeddings for text chunks using sentence transformers.
"""

from typing import List, Dict, Optional
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers not available")


class Embedder:
    """Generate embeddings for text chunks."""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 32,
                 device: str = "cpu",
                 normalize_embeddings: bool = True):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embeddings")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def embed(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, any]], show_progress: bool = True) -> List[Dict[str, any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            show_progress: Show progress bar
            
        Returns:
            List of chunks with added 'embedding' field
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed(texts, show_progress=show_progress)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

