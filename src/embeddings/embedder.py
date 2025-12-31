"""
Embedding Generation Module
Creates vector embeddings for text chunks using sentence transformers.
"""

from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    torch = None
    logger.error("sentence-transformers not available")

# Try to import GPU optimizer
try:
    from ..core.gpu_optimizer import get_gpu_optimizer
    GPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZER_AVAILABLE = False


class Embedder:
    """Generate embeddings for text chunks."""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 32,
                 device: str = "cpu",
                 normalize_embeddings: bool = True,
                 enable_mixed_precision: bool = True,
                 enable_pipelining: bool = False):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embeddings")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_pipelining = enable_pipelining
        
        # Initialize GPU optimizer if available
        self.gpu_optimizer = None
        if GPU_OPTIMIZER_AVAILABLE and device in ["mps", "cuda"]:
            try:
                self.gpu_optimizer = get_gpu_optimizer(
                    device=device,
                    enable_mixed_precision=enable_mixed_precision,
                    max_batch_size=batch_size * 2
                )
                logger.info(f"GPU optimizer initialized for {device}")
            except Exception as e:
                logger.debug(f"GPU optimizer not available: {e}")
        
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            
            # #region agent log
            log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
            try:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    import json as json_lib
                    import time
                    import torch
                    actual_device = "unknown"
                    if hasattr(self.model, '_modules') and len(self.model._modules) > 0:
                        first_module = list(self.model._modules.values())[0]
                        if hasattr(first_module, 'device'):
                            actual_device = str(first_module.device)
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "embedder.py:36",
                        "message": "Embedder model loaded",
                        "data": {
                            "model_name": model_name,
                            "requested_device": device,
                            "actual_device": actual_device,
                            "mps_available": torch.backends.mps.is_available() if torch else False,
                            "gpu_optimizer_available": GPU_OPTIMIZER_AVAILABLE,
                            "batch_size": batch_size,
                            "enable_mixed_precision": enable_mixed_precision
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    log_file.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            
            # Optimize model for inference if GPU optimizer available
            if self.gpu_optimizer:
                self.model = self.gpu_optimizer.optimize_model_for_inference(self.model)
            
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
        
        # #region agent log
        log_path = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                import time
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "embedder.py:57",
                    "message": "Before embedding generation",
                    "data": {
                        "num_texts": len(texts),
                        "device": self.device,
                        "batch_size": self.batch_size,
                        "enable_mixed_precision": self.enable_mixed_precision,
                        "gpu_optimizer_available": self.gpu_optimizer is not None
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Use GPU optimizer if available
        if self.gpu_optimizer:
            embeddings = self.gpu_optimizer.batch_encode_optimized(
                self.model,
                texts,
                batch_size=self.batch_size
            )
        else:
            # Fallback to standard encoding
            use_mixed_precision = False
            if self.enable_mixed_precision and torch and self.device in ["mps", "cuda"]:
                try:
                    use_mixed_precision = True
                except:
                    pass
            
            if use_mixed_precision and torch:
                with torch.inference_mode():
                    with torch.autocast(device_type=self.device):
                        embeddings = self.model.encode(
                            texts,
                            batch_size=self.batch_size,
                            show_progress_bar=show_progress,
                            normalize_embeddings=self.normalize_embeddings,
                            convert_to_numpy=True
                        )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True
                )
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                import json as json_lib
                import time
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "embedder.py:95",
                    "message": "After embedding generation",
                    "data": {
                        "embeddings_shape": list(embeddings.shape) if hasattr(embeddings, 'shape') else "unknown",
                        "used_gpu_optimizer": self.gpu_optimizer is not None,
                        "used_mixed_precision": use_mixed_precision if not self.gpu_optimizer else None
                    },
                    "timestamp": int(time.time() * 1000)
                }
                log_file.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        
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

