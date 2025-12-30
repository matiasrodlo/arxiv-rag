"""
Retrieval Module
Implements hybrid search (semantic + keyword) with re-ranking.
"""

from typing import List, Dict, Optional
import numpy as np
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available for re-ranking")


class Retriever:
    """Hybrid retrieval system with semantic and keyword search."""
    
    def __init__(self,
                 vector_store,
                 embedder,
                 use_hybrid_search: bool = True,
                 hybrid_alpha: float = 0.7,
                 use_reranking: bool = True,
                 rerank_top_k: int = 50,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_hybrid_search = use_hybrid_search and BM25_AVAILABLE
        self.hybrid_alpha = hybrid_alpha
        self.use_reranking = use_reranking and CROSS_ENCODER_AVAILABLE
        self.rerank_top_k = rerank_top_k
        
        # Initialize re-ranker
        if self.use_reranking:
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info(f"Loaded re-ranker model: {reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load re-ranker: {e}")
                self.use_reranking = False
                self.reranker = None
        else:
            self.reranker = None
        
        # BM25 index (built on-the-fly from vector store if needed)
        self.bm25_index = None
        self.bm25_texts = None
    
    def search(self,
               query: str,
               top_k: int = 10,
               filter_metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant chunks with scores
        """
        # Get query embedding
        query_embedding = self.embedder.embed([query], show_progress=False)[0].tolist()
        
        # Semantic search
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.rerank_top_k if self.use_reranking else top_k,
            filter_metadata=filter_metadata
        )
        
        # Hybrid search: combine semantic and keyword
        if self.use_hybrid_search and semantic_results:
            hybrid_results = self._hybrid_search(query, semantic_results, top_k)
        else:
            hybrid_results = semantic_results[:top_k]
        
        # Re-ranking
        if self.use_reranking and hybrid_results:
            reranked_results = self._rerank(query, hybrid_results, top_k)
            return reranked_results
        
        return hybrid_results[:top_k]
    
    def _hybrid_search(self, query: str, semantic_results: List[Dict], top_k: int) -> List[Dict]:
        """Combine semantic and keyword search scores."""
        if not semantic_results:
            return []
        
        # Build BM25 index from semantic results
        texts = [r['text'] for r in semantic_results]
        tokenized_texts = [text.lower().split() for text in texts]
        tokenized_query = query.lower().split()
        
        try:
            bm25 = BM25Okapi(tokenized_texts)
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Normalize scores to [0, 1]
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}. Using semantic only.")
            return semantic_results[:top_k]
        
        # Combine scores
        combined_results = []
        for i, result in enumerate(semantic_results):
            semantic_score = result.get('score', 0.0) or 0.0
            keyword_score = float(bm25_scores[i])
            
            # Weighted combination
            combined_score = (self.hybrid_alpha * semantic_score + 
                            (1 - self.hybrid_alpha) * keyword_score)
            
            combined_results.append({
                **result,
                'score': combined_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:top_k]
    
    def _rerank(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Re-rank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:top_k]
        
        # Prepare query-document pairs
        pairs = [[query, result['text']] for result in results]
        
        # Get re-ranking scores
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}. Returning original results.")
            return results[:top_k]
        
        # Update scores and sort
        for result, score in zip(results, rerank_scores):
            result['rerank_score'] = float(score)
            result['score'] = float(score)  # Use re-rank score as final score
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

