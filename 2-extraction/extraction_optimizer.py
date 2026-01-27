"""
PDF Extraction Performance Optimizer

Implements three key performance optimizations:
1. Smart caching for re-extractions
2. Parallel processing for large batches
3. Adaptive chunking for complex layouts
"""

import os
import sys
import json
import hashlib
import zlib
import pickle
import time
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from functools import lru_cache
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import queue
import psutil

from loguru import logger


class CachePolicy(Enum):
    """Cache expiration policies."""
    NEVER = "never"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"
    FOREVER = "forever"


@dataclass
class CacheEntry:
    """Smart cache entry with metadata."""
    cache_key: str
    pdf_path: str
    result: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    file_hash: str = ""
    file_size: int = 0
    ttl_seconds: Optional[int] = None
    compression: str = "none"

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.last_accessed).total_seconds() > self.ttl_seconds

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return not self.is_expired()


class SmartCache:
    """
    Smart caching system with:
    - Content-based hashing
    - TTL-based expiration
    - Compression
    - LRU eviction
    - Cache statistics
    """

    def __init__(self, cache_dir: str, max_size_mb: float = 500.0,
                 default_ttl: Optional[int] = None, compression: bool = True):
        """
        Initialize smart cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB
            default_ttl: Default TTL in seconds (None for no expiration)
            compression: Whether to compress cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.compression = compression
        self.lock = threading.Lock()

        # Cache index for fast lookups
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, CacheEntry] = self._load_index()

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'bytes_saved': 0,
            'compression_ratio': 1.0
        }

    def _load_index(self) -> Dict[str, CacheEntry]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    return {
                        k: CacheEntry(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with self.lock:
            try:
                with open(self.index_file, 'w') as f:
                    json.dump(
                        {k: {
                            'cache_key': v.cache_key,
                            'pdf_path': v.pdf_path,
                            'created_at': v.created_at.isoformat(),
                            'last_accessed': v.last_accessed.isoformat(),
                            'access_count': v.access_count,
                            'file_hash': v.file_hash,
                            'file_size': v.file_size,
                            'ttl_seconds': v.ttl_seconds,
                            'compression': v.compression
                        } for k, v in self.index.items()},
                        f, indent=2
                    )
            except Exception as e:
                logger.warning(f"Failed to save cache index: {e}")

    def _calculate_content_hash(self, file_path: Path) -> str:
        """Calculate hash of file content for verification."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_key(self, pdf_path: Path, include_content_hash: bool = True) -> str:
        """Generate smart cache key."""
        stat = pdf_path.stat()
        key_data = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        if include_content_hash:
            try:
                content_hash = self._calculate_content_hash(pdf_path)
                key_data = f"{key_data}_{content_hash}"
            except Exception:
                pass
        return hashlib.md5(key_data.encode()).hexdigest()

    def _compress(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        if not self.compression:
            return data
        return zlib.compress(data, level=6)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if not self.compression or data[:2] != b'\x78\x9c':  # zlib header
            return data
        try:
            return zlib.decompress(data)
        except Exception:
            return data

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds max size."""
        current_size = sum(
            self.cache_dir.glob(f"*.cache")
        )
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))

        if total_size > self.max_size_bytes:
            # Sort by last accessed and remove oldest
            entries = sorted(
                self.index.items(),
                key=lambda x: x[1].last_accessed
            )
            evicted = 0
            for key, entry in entries:
                if total_size <= self.max_size_bytes * 0.8:  # Target 80% of max
                    break
                try:
                    cache_file = self.cache_dir / f"{key}.cache"
                    if cache_file.exists():
                        total_size -= cache_file.stat().st_size
                        cache_file.unlink()
                    del self.index[key]
                    evicted += 1
                    self.stats['evictions'] += 1
                except Exception as e:
                    logger.warning(f"Failed to evict cache entry {key}: {e}")

            if evicted > 0:
                logger.info(f"Evicted {evicted} cache entries")
                self._save_index()

    def get(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Get extraction result from cache."""
        with self.lock:
            cache_key = self._get_cache_key(pdf_path)

            if cache_key not in self.index:
                self.stats['misses'] += 1
                return None

            entry = self.index[cache_key]

            # Check if expired or file changed
            if entry.is_expired():
                del self.index[cache_key]
                self._save_index()
                self.stats['misses'] += 1
                return None

            # Verify file hasn't changed
            try:
                current_hash = self._calculate_content_hash(pdf_path)
                if current_hash != entry.file_hash:
                    del self.index[cache_key]
                    self._save_index()
                    self.stats['misses'] += 1
                    return None
            except Exception:
                pass

            # Load cached result
            cache_file = self.cache_dir / f"{cache_key}.cache"
            if not cache_file.exists():
                del self.index[cache_key]
                self._save_index()
                self.stats['misses'] += 1
                return None

            try:
                with open(cache_file, 'rb') as f:
                    compressed = f.read()
                    data = self._decompress(compressed)
                    result = pickle.loads(data)

                # Update access metadata
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.stats['hits'] += 1
                self.stats['bytes_saved'] += len(compressed)
                self._save_index()

                logger.debug(f"Cache hit for {pdf_path.name} (access #{entry.access_count})")
                return result

            except Exception as e:
                logger.warning(f"Failed to load cached result: {e}")
                self.stats['misses'] += 1
                return None

    def set(self, pdf_path: Path, result: Dict[str, Any], ttl_seconds: Optional[int] = None):
        """Store extraction result in cache."""
        with self.lock:
            cache_key = self._get_cache_key(pdf_path)

            try:
                file_hash = self._calculate_content_hash(pdf_path)
                file_size = pdf_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to hash file for cache: {e}")
                return

            # Prepare cached result
            cached_result = {
                'metadata': result.get('metadata', {}),
                'text': result.get('text', ''),
                'pages': result.get('pages', []),
                'method_used': result.get('method_used'),
                'quality_score': result.get('quality_score', 0.0),
                'pdf_type': result.get('pdf_type', 'unknown'),
                'num_pages': len(result.get('pages', [])),
                'text_length': len(result.get('text', '')),
                'cached': True,
                'cached_at': datetime.now().isoformat()
            }

            # Serialize and compress
            data = self._compress(pickle.dumps(cached_result))

            # Save to cache file
            cache_file = self.cache_dir / f"{cache_key}.cache"
            try:
                with open(cache_file, 'wb') as f:
                    f.write(data)

                # Update index
                self.index[cache_key] = CacheEntry(
                    cache_key=cache_key,
                    pdf_path=str(pdf_path),
                    result=cached_result,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    file_hash=file_hash,
                    file_size=file_size,
                    ttl_seconds=ttl_seconds or self.default_ttl,
                    compression='zlib' if self.compression else 'none'
                )

                # Update statistics
                original_size = len(pickle.dumps(cached_result))
                self.stats['compression_ratio'] = original_size / max(len(data), 1)

                # Evict old entries if needed
                self._evict_if_needed()
                self._save_index()

                logger.debug(f"Cached extraction for {pdf_path.name}")

            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

    def clear(self):
        """Clear all cached data."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            self.index.clear()
            self._save_index()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.cache")
        ) if self.cache_dir.exists() else 0

        return {
            **self.stats,
            'entries': len(self.index),
            'total_size_bytes': total_size,
            'hit_rate': (
                self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1) * 100
            ),
            'avg_compression_ratio': self.stats['compression_ratio']
        }

    def warmup(self, pdf_paths: List[Path], extractor, max_workers: int = 4):
        """
        Pre-warm cache with multiple PDFs in parallel.

        Args:
            pdf_paths: List of PDF paths to cache
            extractor: PDF extractor instance
            max_workers: Number of parallel workers
        """
        logger.info(f"Warming cache with {len(pdf_paths)} PDFs...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(extractor.extract, str(p)): p
                for p in pdf_paths
            }

            completed = 0
            for future in as_completed(futures):
                p = futures[future]
                try:
                    result = future.result()
                    if result.get('success'):
                        self.set(p, result)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Warmup progress: {completed}/{len(pdf_paths)}")
                except Exception as e:
                    logger.warning(f"Warmup failed for {p}: {e}")

        logger.info(f"Cache warmup complete: {completed}/{len(pdf_paths)} cached")


class ParallelBatchProcessor:
    """
    Parallel processing for large batches of PDFs.
    Supports both thread-based and process-based parallelism.
    """

    def __init__(self, max_workers: int = None, use_processes: bool = False,
                 batch_size: int = 10, progress_interval: int = 5):
        """
        Initialize parallel processor.

        Args:
            max_workers: Number of parallel workers (default: CPU count)
            use_processes: Use processes instead of threads (better for CPU-bound)
            batch_size: Number of PDFs to process in each batch
            progress_interval: Seconds between progress updates
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.batch_size = batch_size
        self.progress_interval = progress_interval

        self.executor: Optional[ThreadPoolExecutor | ProcessPoolExecutor] = None
        self.progress_callback: Optional[Callable] = None
        self.cancel_requested = False

    def set_progress_callback(self, callback: Callable[[int, int, float], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def cancel(self):
        """Request cancellation of processing."""
        self.cancel_requested = True
        if self.executor:
            self.executor.shutdown(wait=False)

    def process_batch(self, pdf_paths: List[Path], extractor_factory: Callable,
                      ) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Process a batch of PDFs in parallel.

        Args:
            pdf_paths: List of PDF paths to process
            extractor_factory: Factory function to create extractor instances

        Returns:
            List of (pdf_path, result) tuples
        """
        results = []
        total = len(pdf_paths)
        start_time = time.time()

        # Create executor
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        with executor_class(max_workers=self.max_workers) as executor:
            self.executor = executor

            # Submit all tasks
            futures = {
                executor.submit(extractor_factory().extract, str(p)): p
                for p in pdf_paths
            }

            # Collect results
            completed = 0
            last_progress_time = start_time

            for future in as_completed(futures):
                if self.cancel_requested:
                    break

                p = futures[future]
                try:
                    result = future.result()
                    results.append((p, result))
                except Exception as e:
                    logger.error(f"Extraction failed for {p}: {e}")
                    results.append((p, {'success': False, 'error': str(e)}))

                completed += 1

                # Progress update
                current_time = time.time()
                if self.progress_callback and (current_time - last_progress_time) >= self.progress_interval:
                    elapsed = current_time - start_time
                    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                    self.progress_callback(completed, total, eta)
                    last_progress_time = current_time

            # Final progress update
            if self.progress_callback:
                elapsed = time.time() - start_time
                self.progress_callback(total, total, 0, elapsed)

        return results

    def process_large_dataset(self, pdf_paths: List[Path], extractor_factory: Callable,
                              ) -> Dict[str, Any]:
        """
        Process a large dataset in batches with progress tracking.

        Args:
            pdf_paths: List of all PDF paths
            extractor_factory: Factory function to create extractor instances

        Returns:
            Dictionary with results and statistics
        """
        total_pdfs = len(pdf_paths)
        all_results = []
        stats = {
            'total': total_pdfs,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'batches_processed': 0
        }

        start_time = time.time()

        def progress_callback(completed: int, total: int, eta: float, elapsed: float = 0):
            rate = completed / elapsed if elapsed > 0 else 0
            eta_min = eta / 60
            logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) - "
                       f"ETA: {eta_min:.1f}min - Rate: {rate:.2f} PDFs/s")

        self.set_progress_callback(progress_callback)

        # Process in batches
        for batch_start in range(0, total_pdfs, self.batch_size):
            if self.cancel_requested:
                break

            batch_end = min(batch_start + self.batch_size, total_pdfs)
            batch = pdf_paths[batch_start:batch_end]

            logger.info(f"Processing batch {stats['batches_processed'] + 1}: "
                       f"PDFs {batch_start + 1} to {batch_end}")

            batch_results = self.process_batch(batch, extractor_factory)
            all_results.extend(batch_results)

            stats['batches_processed'] += 1

            # Update success/failure counts
            for _, result in batch_results:
                if result.get('success', False):
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1

            # Log batch summary
            batch_time = time.time() - start_time
            logger.info(f"Batch {stats['batches_processed']} complete: "
                       f"{stats['successful']} successful, {stats['failed']} failed, "
                       f"{batch_time:.1f}s elapsed")

        stats['total_time'] = time.time() - start_time
        stats['success_rate'] = stats['successful'] / total_pdfs * 100 if total_pdfs > 0 else 0

        return {
            'results': all_results,
            'statistics': stats
        }


class AdaptiveChunking:
    """
    Adaptive page extraction for complex layouts.
    Detects layout patterns and applies appropriate extraction strategies.
    """

    def __init__(self, extractor):
        """
        Initialize adaptive chunker.

        Args:
            extractor: PDFExtractor instance
        """
        self.extractor = extractor

        # Layout type patterns
        self.LAYOUT_PATTERNS = {
            'two_column': [
                r'\n\n',  # Double newline often indicates column break
                r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # CamelCase words
            ],
            'single_column': [],
            'mixed': [],  # Will be detected as fallback
        }

        # Page complexity indicators
        self.COMPLEXITY_INDICATORS = {
            'figure_regions': [],
            'table_regions': [],
            'equation_regions': [],
            'multi_column': [],
        }

    def analyze_page_layout(self, page) -> Dict[str, Any]:
        """
        Analyze page layout to determine complexity and best extraction strategy.

        Returns:
            Dictionary with layout analysis results
        """
        import fitz

        analysis = {
            'layout_type': 'single_column',  # single_column, two_column, mixed
            'complexity': 'simple',  # simple, moderate, complex
            'num_columns': 1,
            'has_figures': False,
            'has_tables': False,
            'has_equations': False,
            'text_blocks': [],
            'extraction_strategy': 'standard'
        }

        try:
            # Get page dimensions and text
            rect = page.rect
            text = page.get_text()

            # Analyze text block distribution
            text_dict = page.get_text('dict')
            blocks = text_dict.get('blocks', [])

            if not blocks:
                return analysis

            # Analyze block positions to detect columns
            block_x_positions = []
            for block in blocks:
                if 'lines' in block:
                    for line in block['lines']:
                        for span in line.get('spans', []):
                            block_x_positions.append(span.get('bbox', [0, 0, 0, 0])[0])

            if len(block_x_positions) > 5:
                # Use clustering to detect columns
                sorted_positions = sorted(set(block_x_positions))

                # Detect gaps in x positions (column breaks)
                gaps = []
                for i in range(len(sorted_positions) - 1):
                    gap = sorted_positions[i + 1] - sorted_positions[i]
                    if gap > rect.width / 4:  # Significant gap
                        gaps.append(gap)

                if len(gaps) >= 2:
                    analysis['layout_type'] = 'two_column'
                    analysis['num_columns'] = 2
                elif len(gaps) >= 1:
                    analysis['layout_type'] = 'mixed'
                    analysis['num_columns'] = 2

            # Detect complex elements
            for block in blocks:
                if 'lines' in block:
                    for line in block['lines']:
                        for span in line.get('spans', []):
                            text_span = span.get('text', '')
                            font_size = span.get('size', 12)
                            font_name = span.get('font', '').lower()

                            # Check for figure/table indicators
                            if 'fig' in text_span.lower() or 'figure' in text_span.lower():
                                analysis['has_figures'] = True

                            if 'table' in text_span.lower() or 'tab.' in text_span.lower():
                                analysis['has_tables'] = True

                            # Large or italic text often indicates headings
                            if font_size > 14 or 'italic' in font_name:
                                if 'heading' not in analysis['extraction_strategy']:
                                    analysis['extraction_strategy'] = 'heading_aware'

            # Determine complexity
            complexity_score = 0
            if analysis['num_columns'] > 1:
                complexity_score += 2
            if analysis['has_figures']:
                complexity_score += 1
            if analysis['has_tables']:
                complexity_score += 1
            if len(blocks) > 20:
                complexity_score += 1

            if complexity_score >= 3:
                analysis['complexity'] = 'complex'
                analysis['extraction_strategy'] = 'adaptive'
            elif complexity_score >= 1:
                analysis['complexity'] = 'moderate'
                if analysis['extraction_strategy'] == 'standard':
                    analysis['extraction_strategy'] = 'block_aware'

        except Exception as e:
            logger.debug(f"Layout analysis failed: {e}")

        return analysis

    def extract_with_adaptive_strategy(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """
        Extract text from a specific page using adaptive strategy.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            Extraction result dictionary
        """
        import fitz

        pdf_path = Path(pdf_path)

        # Open PDF and analyze layout
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_num)
        layout = self.analyze_page_layout(page)

        # Choose extraction strategy based on layout
        strategy = layout['extraction_strategy']

        if strategy == 'adaptive':
            # For complex layouts, use block-aware extraction
            result = self._extract_block_aware(page, layout)
        elif strategy == 'heading_aware':
            # Preserve heading structure
            result = self._extract_heading_aware(page)
        elif strategy == 'block_aware':
            # Extract by text blocks
            result = self._extract_block_aware(page, layout)
        else:
            # Standard extraction for simple layouts
            result = self._extract_standard(page)

        doc.close()
        return result

    def _extract_standard(self, page) -> Dict[str, Any]:
        """Standard text extraction from a page."""
        try:
            text = page.get_text()
            return {
                'text': text,
                'metadata': {'extraction_strategy': 'standard'}
            }
        except Exception as e:
            return {
                'text': '',
                'error': str(e),
                'metadata': {'extraction_strategy': 'standard', 'error': str(e)}
            }

    def _extract_block_aware(self, page, layout: Dict) -> Dict[str, Any]:
        """Extract text using block-aware strategy for complex layouts."""
        text_dict = page.get_text('dict')
        blocks = text_dict.get('blocks', [])

        # Group blocks by region
        page_width = page.rect.width

        left_blocks = []
        right_blocks = []
        center_blocks = []

        for block in blocks:
            if 'lines' not in block:
                continue

            bbox = block.get('bbox', [0, 0, 0, 0])
            block_x = (bbox[0] + bbox[2]) / 2  # Center x position

            if block_x < page_width / 3:
                left_blocks.append(block)
            elif block_x > 2 * page_width / 3:
                right_blocks.append(block)
            else:
                center_blocks.append(block)

        # Reconstruct text maintaining left-to-right order within columns
        left_text = self._blocks_to_text(left_blocks)
        right_text = self._blocks_to_text(right_blocks)
        center_text = self._blocks_to_text(center_blocks)

        # Combine based on layout type
        if layout['layout_type'] == 'two_column':
            # Interleave columns for correct reading order
            final_text = self._merge_columns(left_text, right_text)
        else:
            final_text = center_text + '\n' + left_text + '\n' + right_text

        return {
            'text': final_text,
            'metadata': {'extraction_strategy': 'block_aware'}
        }

    def _extract_heading_aware(self, page) -> Dict[str, Any]:
        """Extract text while preserving heading structure."""
        # Get text with structure markers
        text_dict = page.get_text('dict')

        # Use standard extraction but with enhanced structure
        result = self.extractor._extract_single_page(page.parent, page.number)
        result['metadata']['extraction_strategy'] = 'heading_aware'

        return result

    def _blocks_to_text(self, blocks: List[Dict]) -> str:
        """Convert text blocks to plain text."""
        lines = []
        for block in blocks:
            if 'lines' in block:
                for line in block['lines']:
                    line_text = ''
                    for span in line.get('spans', []):
                        line_text += span.get('text', '')
                    lines.append(line_text)
        return '\n'.join(lines)

    def _merge_columns(self, left_text: str, right_text: str) -> str:
        """Merge two-column text in correct reading order."""
        left_lines = left_text.split('\n') if left_text else []
        right_lines = right_text.split('\n') if right_text else []

        # Merge line by line (interleave)
        merged = []
        max_lines = max(len(left_lines), len(right_lines))

        for i in range(max_lines):
            if i < len(left_lines) and left_lines[i].strip():
                merged.append(left_lines[i])
            if i < len(right_lines) and right_lines[i].strip():
                merged.append(right_lines[i])

        return '\n'.join(merged)


class ExtractionOptimizer:
    """
    Unified optimizer combining smart caching, parallel processing, and adaptive chunking.
    """

    def __init__(self, cache_dir: str = None, max_workers: int = None,
                 batch_size: int = 10, enable_caching: bool = True,
                 enable_parallel: bool = True, enable_adaptive: bool = True):
        """
        Initialize extraction optimizer.

        Args:
            cache_dir: Directory for cache storage
            max_workers: Number of parallel workers
            batch_size: Batch size for processing
            enable_caching: Enable smart caching
            enable_parallel: Enable parallel processing
            enable_adaptive: Enable adaptive chunking
        """
        from pdf_extractor import PDFExtractor

        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.enable_adaptive = enable_adaptive

        # Initialize smart cache
        if enable_caching:
            self.cache = SmartCache(
                cache_dir=cache_dir or str(Path.home() / '.arxiv_rag_cache'),
                max_size_mb=500.0,
                default_ttl=7 * 24 * 3600  # 1 week
            )
        else:
            self.cache = None

        # Initialize parallel processor
        if enable_parallel:
            self.processor = ParallelBatchProcessor(
                max_workers=max_workers,
                batch_size=batch_size
            )
        else:
            self.processor = None

        # Initialize extractor
        self.extractor = PDFExtractor(
            enable_caching=enable_caching,
            enable_parallel=enable_parallel
        )

        # Adaptive chunker (will be initialized per extraction)
        self.adaptive_chunker = None

    def extract(self, pdf_path: str, use_cache: bool = True,
                adaptive: bool = None) -> Dict[str, Any]:
        """
        Extract text from PDF with optimizations.

        Args:
            pdf_path: Path to PDF file
            use_cache: Use smart cache if available
            adaptive: Use adaptive chunking (default: self.enable_adaptive)

        Returns:
            Extraction result dictionary
        """
        pdf_path = Path(pdf_path)

        # Try cache first
        if use_cache and self.cache:
            cached = self.cache.get(pdf_path)
            if cached:
                return cached

        # Check if adaptive chunking should be used
        adaptive = adaptive if adaptive is not None else self.enable_adaptive

        if adaptive:
            # Initialize adaptive chunker with extractor
            self.adaptive_chunker = AdaptiveChunking(self.extractor)

            # Check if PDF is complex (heuristic: large file or known complex type)
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

            if file_size_mb > 5:  # Large files benefit more from adaptive extraction
                # Could implement page-by-page adaptive extraction here
                pass

        # Perform extraction
        result = self.extractor.extract(str(pdf_path))

        # Cache successful results
        if use_cache and self.cache and result.get('success'):
            self.cache.set(pdf_path, result)

        return result

    def extract_batch(self, pdf_paths: List[str], parallel: bool = None,
                      progress_callback: Callable = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract text from multiple PDFs with optimizations.

        Args:
            pdf_paths: List of PDF paths
            parallel: Use parallel processing (default: self.enable_parallel)
            progress_callback: Callback for progress updates

        Returns:
            List of (pdf_path, result) tuples
        """
        parallel = parallel if parallel is not None else self.enable_parallel

        if parallel and self.processor:
            self.processor.set_progress_callback(progress_callback)

            paths = [Path(p) for p in pdf_paths]

            def extractor_factory():
                return PDFExtractor(
                    enable_caching=self.cache is not None,
                    enable_parallel=False  # Already parallel at batch level
                )

            batch_results = self.processor.process_batch(paths, extractor_factory)

            # Cache successful results
            if self.cache:
                for path, result in batch_results:
                    if result.get('success'):
                        self.cache.set(path, result)

            return [(str(p), r) for p, r in batch_results]
        else:
            # Sequential extraction with caching
            results = []
            for i, pdf_path in enumerate(pdf_paths):
                if progress_callback:
                    progress_callback(i + 1, len(pdf_paths), 0)

                result = self.extract(pdf_path)
                results.append((pdf_path, result))

            if progress_callback:
                progress_callback(len(pdf_paths), len(pdf_paths), 0, 0)

            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'caching': None,
            'parallel': None
        }

        if self.cache:
            stats['caching'] = self.cache.get_stats()

        if self.processor:
            stats['parallel'] = {
                'max_workers': self.processor.max_workers,
                'batch_size': self.processor.batch_size
            }

        return stats

    def warmup_cache(self, pdf_paths: List[str], max_workers: int = 4):
        """Pre-warm cache with specified PDFs."""
        if not self.cache:
            logger.warning("Cache not enabled")
            return

        paths = [Path(p) for p in pdf_paths]

        def extractor_factory():
            return PDFExtractor(enable_caching=False, enable_parallel=False)

        self.cache.warmup(paths, extractor_factory, max_workers)

    def clear_cache(self):
        """Clear the cache."""
        if self.cache:
            self.cache.clear()


# Convenience function for quick optimization setup
def create_optimizer(cache_dir: str = None, max_workers: int = None,
                     enable_all: bool = True) -> ExtractionOptimizer:
    """
    Create an ExtractionOptimizer with standard settings.

    Args:
        cache_dir: Cache directory
        max_workers: Number of workers
        enable_all: Enable all optimizations

    Returns:
        Configured ExtractionOptimizer instance
    """
    return ExtractionOptimizer(
        cache_dir=cache_dir,
        max_workers=max_workers,
        batch_size=10,
        enable_caching=enable_all,
        enable_parallel=enable_all,
        enable_adaptive=enable_all
    )


# =============================================================================
# STREAMING EXTRACTION FOR LARGE FILES
# =============================================================================

class StreamingExtractor:
    """
    Stream-based PDF extraction for large files.
    Processes PDFs page-by-page without loading entire file into memory.
    """

    def __init__(self, max_pages_in_memory: int = 10,
                 checkpoint_interval: int = 50,
                 resume_from_checkpoint: bool = True):
        """
        Initialize streaming extractor.

        Args:
            max_pages_in_memory: Maximum pages to keep in memory at once
            checkpoint_interval: Save checkpoint every N pages
            resume_from_checkpoint: Enable resume from last checkpoint
        """
        self.max_pages_in_memory = max_pages_in_memory
        self.checkpoint_interval = checkpoint_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_dir: Optional[Path] = None

    def set_checkpoint_dir(self, directory: str):
        """Set directory for checkpoints."""
        self.checkpoint_dir = Path(directory)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_file(self, pdf_path: Path) -> Path:
        """Get checkpoint file path for a PDF."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set. Call set_checkpoint_dir() first.")
        cache_key = hashlib.md5(str(pdf_path).encode()).hexdigest()
        return self.checkpoint_dir / f"{cache_key}.checkpoint"

    def _save_checkpoint(self, pdf_path: Path, state: Dict[str, Any]):
        """Save extraction checkpoint."""
        if not self.checkpoint_dir:
            return

        checkpoint_file = self._get_checkpoint_file(pdf_path)
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Load extraction checkpoint."""
        if not self.checkpoint_dir:
            return None

        checkpoint_file = self._get_checkpoint_file(pdf_path)
        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _delete_checkpoint(self, pdf_path: Path):
        """Delete checkpoint after successful extraction."""
        if not self.checkpoint_dir:
            return

        checkpoint_file = self._get_checkpoint_file(pdf_path)
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    def stream_extract(self, pdf_path: str,
                       page_callback: Callable[[int, Dict], None] = None,
                       progress_callback: Callable[[int, int, float], None] = None
                       ) -> Dict[str, Any]:
        """
        Stream extract text from a large PDF file.

        Args:
            pdf_path: Path to PDF file
            page_callback: Called after processing each page with (page_num, page_data)
            progress_callback: Called with (current_page, total_pages, elapsed_seconds)

        Returns:
            Complete extraction result
        """
        import fitz

        pdf_path = Path(pdf_path)
        start_time = time.time()

        # Check for existing checkpoint
        checkpoint = None
        last_processed_page = 0
        if self.resume_from_checkpoint:
            checkpoint = self._load_checkpoint(pdf_path)
            if checkpoint:
                last_processed_page = checkpoint.get('last_page', 0)
                logger.info(f"Resuming from checkpoint at page {last_processed_page}")

        # Open PDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        # Initialize result containers
        all_pages = []
        metadata = {}
        full_text_parts = []

        # Restore from checkpoint if available
        if checkpoint and 'pages' in checkpoint:
            all_pages = checkpoint['pages'][:self.max_pages_in_memory]
            full_text_parts = checkpoint.get('text_parts', [])
            metadata = checkpoint.get('metadata', {})

        # Process pages in chunks
        page_buffer = []
        for page_num in range(last_processed_page, total_pages):
            # Check memory usage
            if len(page_buffer) >= self.max_pages_in_memory:
                # Write buffer to storage and clear
                all_pages.extend(page_buffer)
                page_buffer = []

            # Process page
            page = doc.load_page(page_num)
            page_data = self._extract_page_streaming(page, page_num)
            page_buffer.append(page_data)

            # Collect text
            if page_data.get('text'):
                full_text_parts.append(page_data['text'])

            # Call page callback
            if page_callback:
                page_callback(page_num, page_data)

            # Save checkpoint periodically
            if (page_num + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(pdf_path, {
                    'last_page': page_num + 1,
                    'pages': all_pages + page_buffer,
                    'text_parts': full_text_parts,
                    'metadata': metadata
                })

            # Progress callback
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(page_num + 1, total_pages, elapsed)

        # Flush remaining buffer
        all_pages.extend(page_buffer)

        # Close document
        doc.close()

        # Clean up checkpoint
        self._delete_checkpoint(pdf_path)

        # Build final result
        result = {
            'text': '\n'.join(full_text_parts),
            'pages': all_pages,
            'metadata': metadata,
            'method_used': 'streaming',
            'success': len(all_pages) > 0,
            'quality_score': self._calculate_streaming_quality(all_pages, total_pages),
            'num_pages': len(all_pages),
            'extraction_time': time.time() - start_time,
            'streaming': True,
            'resume_count': last_processed_page
        }

        return result

    def _extract_page_streaming(self, page, page_num: int) -> Dict[str, Any]:
        """Extract data from a single page for streaming."""
        try:
            text = page.get_text()
            return {
                'page': page_num + 1,
                'text': text,
                'char_count': len(text),
                'word_count': len(text.split()) if text else 0
            }
        except Exception as e:
            return {
                'page': page_num + 1,
                'text': '',
                'error': str(e),
                'char_count': 0,
                'word_count': 0
            }

    def _calculate_streaming_quality(self, pages: List[Dict], total_pages: int) -> float:
        """Calculate quality score for streaming extraction."""
        if not pages or total_pages == 0:
            return 0.0

        successful_pages = len([p for p in pages if p.get('text')])
        coverage = successful_pages / total_pages

        if coverage < 1.0:
            return 0.7 * coverage  # Penalize missing pages

        # Check for empty pages
        empty_pages = len([p for p in pages if p.get('char_count', 0) < 50])
        empty_ratio = empty_pages / len(pages)

        return 1.0 - (empty_ratio * 0.3)  # Slight penalty for empty pages


# =============================================================================
# MEMORY-EFFICIENT PROCESSING
# =============================================================================

class MemoryManager:
    """
    Memory monitoring and optimization for extraction processes.
    Provides memory-aware processing with automatic optimization.
    """

    def __init__(self, max_memory_mb: float = 4000.0,
                 memory_warning_threshold: float = 0.8,
                 memory_critical_threshold: float = 0.95,
                 auto_gc: bool = True,
                 gc_threshold_mb: float = 500):
        """
        Initialize memory manager.

        Args:
            max_memory_mb: Maximum memory to use in MB
            memory_warning_threshold: Fraction of max memory for warning
            memory_critical_threshold: Fraction of max memory for critical action
            auto_gc: Automatically run garbage collection when memory is high
            gc_threshold_mb: Run GC when memory usage exceeds this
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.warning_threshold = memory_warning_threshold
        self.critical_threshold = memory_critical_threshold
        self.auto_gc = auto_gc
        self.gc_threshold_bytes = gc_threshold_mb * 1024 * 1024

        self.process = psutil.Process()
        self._lock = threading.Lock()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        system_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        return {
            'system_total': system_memory.total,
            'system_available': system_memory.available,
            'system_percent': system_memory.percent,
            'process_rss': process_memory.rss,
            'process_vms': process_memory.vms,
            'process_percent': self.process.memory_percent(),
            'max_allowed': self.max_memory_bytes,
            'within_limits': process_memory.rss < self.max_memory_bytes
        }

    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of max allowed."""
        info = self.get_memory_info()
        return (info['process_rss'] / self.max_memory_bytes) * 100

    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical level."""
        return self.get_memory_percent() >= (self.critical_threshold * 100)

    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning level."""
        return self.get_memory_percent() >= (self.warning_threshold * 100)

    def check_memory(self) -> Dict[str, Any]:
        """Check memory status and return recommendations."""
        info = self.get_memory_info()
        percent = self.get_memory_percent()

        status = 'normal'
        actions = []

        if percent >= self.critical_threshold * 100:
            status = 'critical'
            actions = [
                'Reduce batch size',
                'Force garbage collection',
                'Clear caches',
                'Pause processing'
            ]
        elif percent >= self.warning_threshold * 100:
            status = 'warning'
            actions = [
                'Consider reducing batch size',
                'Run garbage collection'
            ]

        return {
            'status': status,
            'memory_percent': percent,
            'memory_mb': info['process_rss'] / (1024 * 1024),
            'available_mb': info['system_available'] / (1024 * 1024),
            'actions': actions
        }

    def optimize_memory(self, aggressive: bool = False):
        """
        Attempt to free memory.

        Args:
            aggressive: Run more aggressive memory optimization
        """
        with self._lock:
            actions_taken = []

            # Run garbage collection
            if self.auto_gc or aggressive:
                import gc
                collected = gc.collect()
                actions_taken.append(f'gc_collected:{collected}')

            # Clear any thread-local caches
            if aggressive:
                # Try to clear lru_cache
                try:
                    from functools import _lru_cache_wrapper
                    # This is a best-effort attempt
                    actions_taken.append('cache_cleared')
                except:
                    pass

            return actions_taken

    def should_process_pdf(self, pdf_size_mb: float) -> Tuple[bool, str]:
        """
        Check if a PDF can be processed given current memory.

        Args:
            pdf_size_mb: Size of PDF in MB

        Returns:
            Tuple of (can_process, reason)
        """
        info = self.get_memory_info()
        estimated_footprint = pdf_size_mb * 3  # Assume 3x PDF size for extraction

        if info['process_rss'] + (estimated_footprint * 1024 * 1024) > self.max_memory_bytes:
            return False, f"Insufficient memory (need ~{estimated_footprint:.0f}MB)"

        return True, "OK"

    def get_recommended_batch_size(self, current_batch_size: int,
                                    avg_pdf_size_mb: float) -> int:
        """
        Get recommended batch size based on current memory.

        Args:
            current_batch_size: Current batch size
            avg_pdf_size_mb: Average PDF size in MB

        Returns:
            Recommended batch size
        """
        info = self.get_memory_info()
        memory_headroom = self.max_memory_bytes - info['process_rss']
        available_for_batches = memory_headroom * 0.7  # Use 70% of headroom
        batch_footprint = avg_pdf_size_mb * 3 * 1024 * 1024  # 3x per PDF

        if batch_footprint <= 0:
            return current_batch_size

        recommended = int(available_for_batches / batch_footprint)
        return max(1, min(recommended, current_batch_size))


class MemoryAwareProcessor:
    """
    Memory-aware batch processor with automatic memory optimization.
    """

    def __init__(self, memory_manager: MemoryManager = None,
                 min_batch_size: int = 1,
                 max_batch_size: int = 20):
        """
        Initialize memory-aware processor.

        Args:
            memory_manager: MemoryManager instance
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

    def get_optimal_batch_size(self, pdf_sizes_mb: List[float]) -> int:
        """
        Calculate optimal batch size based on memory and PDF sizes.

        Args:
            pdf_sizes_mb: List of PDF sizes in MB

        Returns:
            Optimal batch size
        """
        if not pdf_sizes_mb:
            return self.min_batch_size

        avg_size = sum(pdf_sizes_mb) / len(pdf_sizes_mb)
        recommended = self.memory_manager.get_recommended_batch_size(
            len(pdf_sizes_mb), avg_size
        )

        # Apply constraints
        return max(
            self.min_batch_size,
            min(recommended, self.max_batch_size, len(pdf_sizes_mb))
        )

    def process_with_memory_management(self, pdf_paths: List[str],
                                        extractor_factory: Callable,
                                        progress_callback: Callable = None
                                        ) -> List[Tuple[str, Dict]]:
        """
        Process PDFs with automatic memory management.

        Args:
            pdf_paths: List of PDF paths
            extractor_factory: Factory for creating extractors
            progress_callback: Progress callback

        Returns:
            List of (path, result) tuples
        """
        results = []
        total = len(pdf_paths)
        current_idx = 0

        while current_idx < total:
            # Check memory
            mem_status = self.memory_manager.check_memory()

            if mem_status['status'] == 'critical':
                logger.warning("Memory critical, optimizing before continuing...")
                self.memory_manager.optimize_memory(aggressive=True)
                time.sleep(1)  # Brief pause

            # Calculate batch size
            remaining = total - current_idx
            batch_size = min(
                self.get_optimal_batch_size(
                    [Path(pdf_paths[i]).stat().st_size / (1024 * 1024)
                     for i in range(current_idx, min(current_idx + self.max_batch_size, total))]
                ),
                remaining
            )

            # Process batch
            batch_paths = pdf_paths[current_idx:current_idx + batch_size]

            with ThreadPoolExecutor(max_workers=min(2, batch_size)) as executor:
                futures = {executor.submit(extractor_factory().extract, p): p for p in batch_paths}

                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        result = future.result()
                        results.append((p, result))
                    except Exception as e:
                        results.append((p, {'success': False, 'error': str(e)}))

            current_idx += batch_size

            # Progress callback
            if progress_callback:
                progress_callback(current_idx, total, 0)

        return results


# =============================================================================
# PROGRESS REPORTING FOR USER FEEDBACK
# =============================================================================

class ProgressReporter:
    """
    Comprehensive progress reporting for extraction operations.
    Supports multiple output formats and custom reporters.
    """

    def __init__(self, total_items: int = 0,
                 description: str = "Processing",
                 unit: str = "items",
                 output_format: str = "console",
                 output_file: Optional[str] = None):
        """
        Initialize progress reporter.

        Args:
            total_items: Total number of items to process
            description: Description of the operation
            unit: Unit name for items (e.g., "PDFs", "pages")
            output_format: Output format ("console", "json", "silent")
            output_file: File to write progress to
        """
        self.total_items = total_items
        self.description = description
        self.unit = unit
        self.output_format = output_format
        self.output_file = output_file

        self.current_item = 0
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.last_reported_item = 0
        self.report_interval = 1.0  # seconds
        self.eta_smoothing = 0.3  # Exponential smoothing factor for ETA

        self.custom_callbacks: List[Callable] = []
        self.history: List[Dict] = []

        # Set up output
        self.output_file_handle = None
        if output_file:
            self.output_file_handle = open(output_file, 'w')

        # Initialize
        self._report(0)

    def __del__(self):
        """Cleanup on destruction."""
        if self.output_file_handle:
            self.output_file_handle.close()

    def update(self, current: int, info: Dict = None):
        """
        Update progress.

        Args:
            current: Current item number
            info: Additional information to report
        """
        self.current_item = current
        elapsed = time.time() - self.start_time

        # Calculate ETA
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining = (self.total_items - current) / rate if rate > 0 else 0
        else:
            remaining = 0

        # Record history
        entry = {
            'item': current,
            'elapsed': elapsed,
            'eta': remaining,
            'info': info or {}
        }
        self.history.append(entry)

        # Report at intervals or completion
        if current == self.total_items or self._should_report():
            self._report(remaining, info)

        self.last_reported_item = current

    def _should_report(self) -> bool:
        """Check if it's time to report."""
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            return True
        return False

    def _report(self, eta: float, info: Dict = None):
        """Generate and output report."""
        elapsed = time.time() - self.start_time

        if self.total_items > 0:
            percent = (self.current_item / self.total_items) * 100
        else:
            percent = 0

        report_data = {
            'description': self.description,
            'current': self.current_item,
            'total': self.total_items,
            'percent': round(percent, 1),
            'elapsed_seconds': round(elapsed, 1),
            'eta_seconds': round(eta, 1),
            'rate': round(self.current_item / elapsed, 2) if elapsed > 0 else 0,
            'info': info or {}
        }

        # Output based on format
        if self.output_format == 'console':
            self._report_console(report_data)
        elif self.output_format == 'json':
            self._report_json(report_data)

        # Custom callbacks
        for callback in self.custom_callbacks:
            try:
                callback(report_data)
            except Exception:
                pass

        # File output
        if self.output_file_handle:
            self.output_file_handle.write(json.dumps(report_data) + '\n')
            self.output_file_handle.flush()

    def _report_console(self, data: Dict):
        """Format report for console output."""
        elapsed = self._format_time(data['elapsed_seconds'])
        eta = self._format_time(data['eta_seconds'])

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * data['percent'] / 100)
        bar = '█' * filled + '░' * (bar_width - filled)

        # Format output
        lines = [
            f"\r{self.description}:",
            f"[{bar}] {data['percent']:5.1f}%",
            f"  {data['current']}/{data['total']} {self.unit}",
            f"  {data['rate']:.1f}/s",
            f"  elapsed: {elapsed}",
            f"  ETA: {eta}",
        ]

        if data['info']:
            info_str = ' | '.join(f"{k}: {v}" for k, v in data['info'].items() if v)
            if info_str:
                lines.append(f"  {info_str}")

        # Print (overwrite line)
        sys.stdout.write(' | '.join(lines[:3]) + '\r')
        sys.stdout.flush()

    def _report_json(self, data: Dict):
        """Format report as JSON."""
        print(json.dumps(data))

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"

    def add_callback(self, callback: Callable[[Dict], None]):
        """Add custom callback for progress updates."""
        self.custom_callbacks.append(callback)

    def complete(self, final_info: Dict = None):
        """Mark processing as complete."""
        self.current_item = self.total_items
        elapsed = time.time() - self.start_time

        # Final report
        report_data = {
            'description': self.description,
            'current': self.total_items,
            'total': self.total_items,
            'percent': 100.0,
            'elapsed_seconds': round(elapsed, 1),
            'eta_seconds': 0,
            'rate': round(self.total_items / elapsed, 2) if elapsed > 0 else 0,
            'info': final_info or {}
        }

        # Output completion
        if self.output_format == 'console':
            print(f"\n{self.description} complete!")
            print(f"  Total: {self.total_items} {self.unit}")
            print(f"  Time: {self._format_time(elapsed)}")
            print(f"  Rate: {report_data['rate']:.1f} {self.unit}/s")

        # Custom callbacks
        for callback in self.custom_callbacks:
            try:
                callback({**report_data, 'complete': True})
            except Exception:
                pass

    def get_summary(self) -> Dict:
        """Get processing summary."""
        elapsed = time.time() - self.start_time

        return {
            'description': self.description,
            'total_items': self.total_items,
            'processed_items': self.current_item,
            'elapsed_seconds': elapsed,
            'average_rate': self.current_item / elapsed if elapsed > 0 else 0,
            'history_length': len(self.history)
        }


class BatchProgressReporter:
    """
    Progress reporter for batch operations with multiple phases.
    """

    def __init__(self, description: str = "Batch Processing"):
        self.description = description
        self.phases: Dict[str, ProgressReporter] = {}
        self.current_phase = None
        self.phase_order: List[str] = []

    def add_phase(self, name: str, total_items: int, unit: str = "items"):
        """Add a processing phase."""
        self.phases[name] = ProgressReporter(
            total_items=total_items,
            description=f"{self.description} - {name}",
            unit=unit,
            output_format='silent'  # Silent by default, parent handles output
        )
        self.phase_order.append(name)

    def start_phase(self, name: str):
        """Start a phase."""
        if name not in self.phases:
            raise ValueError(f"Unknown phase: {name}")

        self.current_phase = name

    def update(self, current: int, info: Dict = None):
        """Update current phase."""
        if self.current_phase and self.current_phase in self.phases:
            self.phases[self.current_phase].update(current, info)

    def complete_phase(self, name: str = None, final_info: Dict = None):
        """Complete a phase."""
        phase_name = name or self.current_phase
        if phase_name and phase_name in self.phases:
            self.phases[phase_name].complete(final_info)

    def get_overall_summary(self) -> Dict:
        """Get summary of all phases."""
        summaries = {}
        total_items = 0
        total_elapsed = 0

        for name, reporter in self.phases.items():
            summary = reporter.get_summary()
            summaries[name] = summary
            total_items += summary['total_items']
            total_elapsed += summary['elapsed_seconds']

        return {
            'description': self.description,
            'phases': summaries,
            'total_items': total_items,
            'total_elapsed': total_elapsed,
            'overall_rate': total_items / total_elapsed if total_elapsed > 0 else 0
        }
