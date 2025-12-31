"""
PDF Pre-loading Cache Manager.
Pre-loads PDFs into memory to eliminate I/O wait time.
"""

import mmap
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict
from loguru import logger
import threading
import time


class PDFCache:
    """
    LRU cache for PDF files in memory.
    Pre-loads PDFs to eliminate disk I/O bottlenecks.
    """
    
    def __init__(self, max_size_mb: int = 10000, preload_batch_size: int = 100):
        """
        Initialize PDF cache.
        
        Args:
            max_size_mb: Maximum cache size in MB (default 10GB for 128GB RAM system)
            preload_batch_size: Number of PDFs to preload ahead
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.preload_batch_size = preload_batch_size
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.cache_sizes: Dict[str, int] = {}
        self.current_size = 0
        self.lock = threading.RLock()
        self.pdf_dir: Optional[Path] = None
        self.hits = 0
        self.misses = 0
        
    def set_pdf_dir(self, pdf_dir: Path):
        """Set the PDF directory to cache from."""
        self.pdf_dir = Path(pdf_dir)
    
    def _load_pdf(self, pdf_path: Path) -> Optional[bytes]:
        """Load PDF file into memory."""
        try:
            if not pdf_path.exists():
                return None
            
            with open(pdf_path, 'rb') as f:
                # Use memory mapping for large files
                if pdf_path.stat().st_size > 10 * 1024 * 1024:  # >10MB
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    data = bytes(mm)
                    mm.close()
                else:
                    data = f.read()
            
            return data
        except Exception as e:
            logger.debug(f"Failed to load PDF {pdf_path}: {e}")
            return None
    
    def get(self, paper_id: str) -> Optional[bytes]:
        """
        Get PDF from cache.
        
        Args:
            paper_id: Paper ID (without .pdf extension)
            
        Returns:
            PDF bytes or None if not in cache
        """
        with self.lock:
            if paper_id in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(paper_id)
                self.hits += 1
                return self.cache[paper_id]
            else:
                self.misses += 1
                return None
    
    def put(self, paper_id: str, pdf_data: bytes):
        """
        Add PDF to cache.
        
        Args:
            paper_id: Paper ID
            pdf_data: PDF file bytes
        """
        with self.lock:
            size = len(pdf_data)
            
            # Remove if already exists
            if paper_id in self.cache:
                self.current_size -= self.cache_sizes[paper_id]
                del self.cache[paper_id]
                del self.cache_sizes[paper_id]
            
            # Evict old entries if needed
            while self.current_size + size > self.max_size_bytes and self.cache:
                # Remove least recently used
                lru_id = next(iter(self.cache))
                self.current_size -= self.cache_sizes[lru_id]
                del self.cache[lru_id]
                del self.cache_sizes[lru_id]
            
            # Add new entry
            self.cache[paper_id] = pdf_data
            self.cache_sizes[paper_id] = size
            self.current_size += size
    
    def preload(self, paper_ids: List[str], background: bool = True):
        """
        Pre-load PDFs into cache.
        
        Args:
            paper_ids: List of paper IDs to preload
            background: If True, preload in background thread
        """
        if not self.pdf_dir:
            logger.warning("PDF directory not set, cannot preload")
            return
        
        def _preload_worker():
            """Background worker to preload PDFs."""
            loaded = 0
            for paper_id in paper_ids:
                if paper_id in self.cache:
                    continue  # Already cached
                
                pdf_path = self.pdf_dir / f"{paper_id}.pdf"
                pdf_data = self._load_pdf(pdf_path)
                if pdf_data:
                    self.put(paper_id, pdf_data)
                    loaded += 1
                    
                    if loaded % 10 == 0:
                        logger.debug(f"Preloaded {loaded} PDFs into cache")
            
            logger.info(f"Preloaded {loaded} PDFs into cache")
        
        if background:
            thread = threading.Thread(target=_preload_worker, daemon=True)
            thread.start()
        else:
            _preload_worker()
    
    def preload_next_batch(self, current_paper_ids: List[str], all_paper_ids: List[str]):
        """
        Preload next batch of papers while current batch is processing.
        
        Args:
            current_paper_ids: Currently processing paper IDs
            all_paper_ids: All paper IDs to process
        """
        # Find current position
        if not current_paper_ids or not all_paper_ids:
            return
        
        try:
            current_idx = all_paper_ids.index(current_paper_ids[0])
            next_batch = all_paper_ids[
                current_idx + len(current_paper_ids):
                current_idx + len(current_paper_ids) + self.preload_batch_size
            ]
            
            if next_batch:
                self.preload(next_batch, background=True)
        except (ValueError, IndexError):
            pass
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            
            return {
                'size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'num_cached': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.cache_sizes.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0


# Global cache instance
_pdf_cache = None

def get_pdf_cache(max_size_mb: int = 10000) -> PDFCache:
    """Get or create global PDF cache."""
    global _pdf_cache
    if _pdf_cache is None:
        _pdf_cache = PDFCache(max_size_mb=max_size_mb)
    return _pdf_cache

