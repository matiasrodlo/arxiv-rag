#!/usr/bin/env python3
"""
PDF Extraction Script

Extracts text from PDFs and saves to /Volumes/8SSD/paper/extracted_texts
with the same folder structure as the source PDFs.

Usage:
    python extract_pdfs_to_disk.py --num-pdfs 100
    python extract_pdfs_to_disk.py --category cs.AI
    python extract_pdfs_to_disk.py --all
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent / '2-extraction'))

from pdf_extractor import PDFExtractor
from extraction_optimizer import (
    SmartCache, StreamingExtractor, MemoryManager,
    ProgressReporter, create_optimizer
)
from loguru import logger


class PDFExtractionSaver:
    """Extract PDFs and save to disk with same folder structure."""

    def __init__(self,
                 source_dir: str = '/Volumes/8SSD/paper/pdfs',
                 output_dir: str = '/Volumes/8SSD/paper/extracted_texts',
                 num_pdfs: Optional[int] = None,
                 seed: int = 42,
                 enable_cache: bool = True,
                 enable_parallel: bool = True,
                 max_workers: int = 4,
                 batch_size: int = 10):
        """
        Initialize the extraction saver.

        Args:
            source_dir: Source directory containing PDFs
            output_dir: Output directory for extracted JSONs
            num_pdfs: Number of PDFs to process (None for all)
            seed: Random seed for reproducibility
            enable_cache: Enable caching of extractions
            enable_parallel: Enable parallel processing
            max_workers: Number of parallel workers
            batch_size: Batch size for processing
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.num_pdfs = num_pdfs
        self.seed = seed
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Initialize components
        self.extractor = PDFExtractor(
            enable_ocr=False,
            enable_caching=enable_cache,
            enable_parallel=enable_parallel,
            max_workers=max_workers
        )

        # Initialize cache
        if enable_cache:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = self.output_dir / '.cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = SmartCache(
                cache_dir=str(cache_dir),
                max_size_mb=500.0,
                default_ttl=7 * 24 * 3600
            )
        else:
            self.cache = None

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            max_memory_mb=3000,
            memory_warning_threshold=0.8,
            memory_critical_threshold=0.95
        )

        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'total_characters': 0,
            'total_pages': 0,
            'cache_hits': 0,
            'errors': []
        }

    def find_pdfs(self, category: Optional[str] = None) -> List[Path]:
        """Find PDFs to process."""
        if category:
            search_dir = self.source_dir / category
        else:
            search_dir = self.source_dir

        if not search_dir.exists():
            raise FileNotFoundError(f"Directory not found: {search_dir}")

        pdfs = list(search_dir.rglob('*.pdf'))

        if self.num_pdfs and len(pdfs) > self.num_pdfs:
            random.seed(self.seed)
            pdfs = random.sample(pdfs, self.num_pdfs)

        self.stats['total_pdfs'] = len(pdfs)
        logger.info(f"Found {len(pdfs)} PDFs to process")
        return sorted(pdfs)

    def get_output_path(self, pdf_path: Path) -> Path:
        """Get output path maintaining folder structure."""
        # Get relative path from source
        relative_path = pdf_path.relative_to(self.source_dir)

        # Create output path (replace .pdf with .json)
        json_filename = relative_path.stem + '.json'
        output_path = self.output_dir / relative_path.parent / json_filename

        return output_path

    def save_extraction(self, pdf_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """Save extraction result to disk."""
        output_path = self.get_output_path(pdf_path)

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        save_data = {
            'pdf_path': str(pdf_path),
            'pdf_name': pdf_path.name,
            'relative_path': str(pdf_path.relative_to(self.source_dir)),
            'file_size_bytes': pdf_path.stat().st_size,
            'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 4),
            'success': result.get('success', False),
            'method_used': result.get('method_used'),
            'pdf_type': result.get('pdf_type', 'unknown'),
            'quality_score': result.get('quality_score', 0.0),
            'extraction_time_seconds': result.get('extraction_time_seconds', 0),
            'num_pages': result.get('num_pages', 0),
            'char_count': result.get('char_count', 0),
            'word_count': result.get('word_count', 0),
            'reading_time_minutes': result.get('reading_time_minutes', 0),
            'metadata': result.get('metadata', {}),
            'extracted_at': datetime.now().isoformat()
        }

        # Include text and pages only for successful extractions
        if result.get('success', False):
            save_data['text'] = result.get('text', '')[:100000]  # Limit text size
            save_data['pages'] = result.get('pages', [])
        else:
            save_data['error'] = result.get('error', 'Unknown error')
            save_data['text'] = ''
            save_data['pages'] = []

        # Save to JSON
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            return {'success': True, 'output_path': output_path}
        except Exception as e:
            return {'success': False, 'error': str(e), 'output_path': output_path}

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF."""
        # Check memory before processing
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        can_process, reason = self.memory_manager.should_process_pdf(file_size_mb)
        if not can_process:
            return {
                'pdf_path': str(pdf_path),
                'success': False,
                'error': f"Memory constraint: {reason}"
            }

        # Try cache first
        if self.cache:
            cached = self.cache.get(pdf_path)
            if cached:
                self.stats['cache_hits'] += 1
                result = cached
                result['cached'] = True
            else:
                result = self.extractor.extract(str(pdf_path))
                if result.get('success'):
                    self.cache.set(pdf_path, result)
        else:
            result = self.extractor.extract(str(pdf_path))

        # Save extraction
        save_result = self.save_extraction(pdf_path, result)

        # Update statistics
        if result.get('success', False):
            self.stats['successful'] += 1
            self.stats['total_characters'] += result.get('char_count', 0)
            self.stats['total_pages'] += result.get('num_pages', 0)
        else:
            self.stats['failed'] += 1
            self.stats['errors'].append({
                'pdf': str(pdf_path),
                'error': result.get('error', 'Unknown')
            })

        return {
            'pdf_path': str(pdf_path),
            'success': result.get('success', False),
            'output_path': str(save_result.get('output_path', '')),
            'error': save_result.get('error', result.get('error'))
        }

    def process_all(self, category: Optional[str] = None,
                    progress_callback=None) -> Dict[str, Any]:
        """
        Process all PDFs.

        Args:
            category: Optional category filter (e.g., 'cs.AI')
            progress_callback: Optional callback(current, total, elapsed_seconds)

        Returns:
            Processing statistics
        """
        self.stats['start_time'] = datetime.now()
        start_time = time.time()

        # Find PDFs
        pdfs = self.find_pdfs(category)
        total = len(pdfs)

        logger.info(f"Processing {total} PDFs...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process with memory-aware batching
        results = []
        current_idx = 0

        while current_idx < total:
            # Check memory
            mem_status = self.memory_manager.check_memory()
            if mem_status['status'] == 'critical':
                logger.warning("Memory critical, pausing...")
                self.memory_manager.optimize_memory(aggressive=True)
                time.sleep(2)

            # Calculate batch size
            remaining = total - current_idx
            batch_size = min(
                self.memory_manager.get_recommended_batch_size(
                    self.batch_size,
                    sum(p.stat().st_size for p in pdfs[current_idx:current_idx + self.batch_size]) / (1024 * 1024) / self.batch_size
                    if current_idx + self.batch_size <= total else 2.5
                ),
                remaining
            )

            batch = pdfs[current_idx:current_idx + batch_size]

            # Process batch
            with ThreadPoolExecutor(max_workers=min(2, batch_size)) as executor:
                futures = {executor.submit(self.process_pdf, p): p for p in batch}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            current_idx += batch_size

            # Progress callback
            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback(current_idx, total, elapsed)
            elif current_idx % 10 == 0:
                logger.info(f"Progress: {current_idx}/{total} ({current_idx/total*100:.1f}%) - "
                           f"{elapsed:.1f}s elapsed")

        self.stats['end_time'] = datetime.now()
        self.stats['processing_time'] = time.time() - start_time

        # Save statistics
        self._save_statistics()

        return self.stats

    def _save_statistics(self):
        """Save processing statistics."""
        stats_path = self.output_dir / 'extraction_statistics.json'

        stats = {
            'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
            'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
            'processing_time_seconds': self.stats.get('processing_time', 0),
            'total_pdfs': self.stats['total_pdfs'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': round(
                self.stats['successful'] / max(self.stats['total_pdfs'], 1) * 100, 2
            ),
            'total_characters': self.stats['total_characters'],
            'total_pages': self.stats['total_pages'],
            'cache_hits': self.stats['cache_hits'],
            'average_pdfs_per_second': round(
                self.stats['total_pdfs'] / max(self.stats.get('processing_time', 1), 0.001), 2
            ),
            'errors': self.stats['errors'][:50]  # Limit errors
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")

    def print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)

        stats = self.stats

        print(f"\n📊 Overview:")
        print(f"   Total PDFs: {stats['total_pdfs']}")
        print(f"   Successful: {stats['successful']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Success Rate: {stats['successful'] / max(stats['total_pdfs'], 1) * 100:.1f}%")

        print(f"\n📄 Content:")
        print(f"   Total Pages: {stats['total_pages']:,}")
        print(f"   Total Characters: {stats['total_characters']:,}")
        print(f"   Cache Hits: {stats['cache_hits']}")

        print(f"\n⏱️  Performance:")
        print(f"   Processing Time: {stats.get('processing_time', 0):.1f}s")
        print(f"   Rate: {stats['total_pdfs'] / max(stats.get('processing_time', 1), 0.001):.2f} PDFs/s")

        print(f"\n💾 Output:")
        print(f"   Directory: {self.output_dir}")
        print(f"   Statistics: {self.output_dir / 'extraction_statistics.json'}")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract text from PDFs and save to disk',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pdfs_to_disk.py --num-pdfs 100
  python extract_pdfs_to_disk.py --category cs.AI
  python extract_pdfs_to_disk.py --all --enable-parallel
  python extract_pdfs_to_disk.py --source /path/to/pdfs --output /path/to/output
        """
    )

    parser.add_argument('--source', '-s',
                        default='/Volumes/8SSD/paper/pdfs',
                        help='Source directory containing PDFs')

    parser.add_argument('--output', '-o',
                        default='/Volumes/8SSD/paper/extracted_texts',
                        help='Output directory for extracted JSONs')

    parser.add_argument('--num-pdfs', '-n',
                        type=int, default=None,
                        help='Number of PDFs to process (default: all)')

    parser.add_argument('--category', '-c',
                        default=None,
                        help='Process specific category (e.g., cs.AI, cs.LG)')

    parser.add_argument('--seed', '-d',
                        type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--no-cache',
                        action='store_true',
                        help='Disable caching')

    parser.add_argument('--parallel', '-p',
                        action='store_true',
                        default=True,
                        help='Enable parallel processing')

    parser.add_argument('--workers', '-w',
                        type=int, default=4,
                        help='Number of parallel workers')

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level,
               format="{time:HH:mm:ss} | {level} | {message}")

    # Create saver and run
    saver = PDFExtractionSaver(
        source_dir=args.source,
        output_dir=args.output,
        num_pdfs=args.num_pdfs,
        seed=args.seed,
        enable_cache=not args.no_cache,
        enable_parallel=args.parallel,
        max_workers=args.workers
    )

    def progress_callback(current, total, elapsed):
        percent = current / total * 100
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        print(f"\rProgress: {current}/{total} ({percent:.1f}%) | "
              f"{rate:.2f} PDFs/s | ETA: {eta/60:.1f}min", end='', flush=True)

    saver.process_all(category=args.category, progress_callback=progress_callback)
    saver.print_summary()

    return 0


if __name__ == '__main__':
    sys.exit(main())
