#!/usr/bin/env python3
"""
Main script to run the RAG pipeline.
Processes ArXiv CS papers and builds the vector database.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline


def setup_logging(config_path: str = "config.yaml"):
    """Setup logging based on configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        log_config = config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('log_file', 'logs/rag_pipeline.log')
        log_rotation = log_config.get('log_rotation', '100 MB')
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            log_file,
            level=log_level,
            rotation=log_rotation,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    except Exception as e:
        logger.warning(f"Failed to load logging config: {e}. Using defaults.")


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline for ArXiv CS papers")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--paper-ids-file',
        type=str,
        default='paper_ids.txt',
        help='Path to file containing paper IDs (default: paper_ids.txt)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (overrides config)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start processing from this index (for resuming)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of papers to process (for testing)'
    )
    parser.add_argument(
        '--paper-id',
        type=str,
        default=None,
        help='Process a single paper by ID'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.config)
    
    logger.info("=" * 60)
    logger.info("Starting RAG Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(config_path=args.config)
        
        if args.paper_id:
            # Process single paper
            logger.info(f"Processing single paper: {args.paper_id}")
            result = pipeline.process_paper(args.paper_id)
            if result:
                logger.info(f"Success: {result}")
            else:
                logger.error(f"Failed to process {args.paper_id}")
                sys.exit(1)
        else:
            # Load paper IDs
            paper_ids_path = Path(args.paper_ids_file)
            if not paper_ids_path.exists():
                logger.error(f"Paper IDs file not found: {args.paper_ids_file}")
                sys.exit(1)
            
            with open(paper_ids_path, 'r') as f:
                paper_ids = [line.strip() for line in f if line.strip()]
            
            # Apply limits
            if args.start_from > 0:
                paper_ids = paper_ids[args.start_from:]
                logger.info(f"Starting from index {args.start_from}")
            
            if args.limit:
                paper_ids = paper_ids[:args.limit]
                logger.info(f"Limiting to {args.limit} papers")
            
            logger.info(f"Processing {len(paper_ids)} papers")
            
            # Process batch
            results = pipeline.process_batch(paper_ids, batch_size=args.batch_size)
            
            # Print statistics
            logger.info("=" * 60)
            logger.info("Processing Complete")
            logger.info("=" * 60)
            logger.info(f"Total papers: {results['total']}")
            logger.info(f"Successful: {results['successful']}")
            logger.info(f"Failed: {results['failed']}")
            
            if results['errors']:
                logger.warning(f"Errors encountered: {len(results['errors'])}")
                for error in results['errors'][:10]:  # Show first 10 errors
                    logger.warning(f"  {error['paper_id']}: {error['error']}")
            
            # Get final statistics
            stats = pipeline.get_stats()
            logger.info("=" * 60)
            logger.info("Vector Store Statistics")
            logger.info("=" * 60)
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

