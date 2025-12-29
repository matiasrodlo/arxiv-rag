#!/usr/bin/env python3
"""
Query script for the RAG system.
Allows querying the vector database and retrieving relevant papers.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument(
        'query',
        type=str,
        help='Search query'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter by category (e.g., cs.LG)'
    )
    parser.add_argument(
        '--paper-id',
        type=str,
        default=None,
        help='Filter by specific paper ID'
    )
    parser.add_argument(
        '--no-rerank',
        action='store_true',
        help='Disable re-ranking'
    )
    parser.add_argument(
        '--no-hybrid',
        action='store_true',
        help='Disable hybrid search (use semantic only)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        embedder = Embedder(
            model_name=config['embeddings']['model'],
            batch_size=config['embeddings']['batch_size'],
            device=config['embeddings']['device'],
            normalize_embeddings=config['embeddings']['normalize_embeddings']
        )
        
        vector_store = VectorStore(
            db_type=config['vector_db']['type'],
            collection_name=config['vector_db']['collection_name'],
            persist_directory=config['vector_db'].get('persist_directory'),
            qdrant_host=config['vector_db'].get('qdrant_host', 'localhost'),
            qdrant_port=config['vector_db'].get('qdrant_port', 6333)
        )
        
        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder,
            use_hybrid_search=not args.no_hybrid and config['retrieval']['use_hybrid_search'],
            hybrid_alpha=config['retrieval']['hybrid_alpha'],
            use_reranking=not args.no_rerank and config['retrieval']['use_reranking'],
            rerank_top_k=config['retrieval']['rerank_top_k'],
            reranker_model=config['retrieval']['reranker_model']
        )
        
        # Build filter
        filter_metadata = {}
        if args.category:
            filter_metadata['categories'] = args.category
        if args.paper_id:
            filter_metadata['paper_id'] = args.paper_id
        
        # Search
        logger.info(f"Querying: {args.query}")
        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        # Display results
        print("\n" + "=" * 80)
        print(f"Query: {args.query}")
        print(f"Found {len(results)} results")
        print("=" * 80 + "\n")
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            paper_id = metadata.get('paper_id', 'unknown')
            title = metadata.get('title', 'N/A')
            score = result.get('score', 0.0)
            
            print(f"{i}. Paper ID: {paper_id}")
            print(f"   Title: {title}")
            print(f"   Score: {score:.4f}")
            if 'semantic_score' in result:
                print(f"   Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f}")
            print(f"   Text preview: {result['text'][:200]}...")
            print()
    
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

