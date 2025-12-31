"""
Advanced optimization techniques for M4 Max.
Combines multiple optimization strategies for maximum performance.
"""

import threading
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from .pdf_cache import get_pdf_cache, PDFCache
from .async_io import get_async_io_helper, AsyncIOHelper


class AdvancedOptimizer:
    """
    Advanced optimizer combining multiple techniques:
    - PDF pre-loading
    - Async I/O
    - Parallel operations
    - Smart batching
    """
    
    def __init__(self, 
                 pdf_cache_size_mb: int = 10000,
                 async_io_workers: int = 8,
                 enable_preloading: bool = True,
                 enable_async_io: bool = True):
        """
        Initialize advanced optimizer.
        
        Args:
            pdf_cache_size_mb: Size of PDF cache in MB
            async_io_workers: Number of async I/O workers
            enable_preloading: Enable PDF pre-loading
            enable_async_io: Enable async I/O operations
        """
        self.enable_preloading = enable_preloading
        self.enable_async_io = enable_async_io
        
        # Initialize PDF cache
        if enable_preloading:
            self.pdf_cache = get_pdf_cache(max_size_mb=pdf_cache_size_mb)
            logger.info(f"PDF cache initialized: {pdf_cache_size_mb}MB max size")
        else:
            self.pdf_cache = None
        
        # Initialize async I/O
        if enable_async_io:
            self.async_io = get_async_io_helper(max_workers=async_io_workers)
            logger.info(f"Async I/O initialized: {async_io_workers} workers")
        else:
            self.async_io = None
    
    def setup_pdf_cache(self, pdf_dir: Path):
        """Setup PDF cache with directory."""
        if self.pdf_cache:
            self.pdf_cache.set_pdf_dir(pdf_dir)
            logger.info(f"PDF cache directory set: {pdf_dir}")
    
    def preload_pdfs(self, paper_ids: List[str], background: bool = True):
        """
        Preload PDFs into memory cache.
        
        Args:
            paper_ids: List of paper IDs to preload
            background: Preload in background thread
        """
        if self.pdf_cache:
            self.pdf_cache.preload(paper_ids, background=background)
    
    def preload_next_batch(self, current_paper_ids: List[str], all_paper_ids: List[str]):
        """Preload next batch while current batch processes."""
        if self.pdf_cache:
            self.pdf_cache.preload_next_batch(current_paper_ids, all_paper_ids)
    
    def get_pdf_from_cache(self, paper_id: str) -> Optional[bytes]:
        """Get PDF from cache if available."""
        if self.pdf_cache:
            return self.pdf_cache.get(paper_id)
        return None
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.pdf_cache:
            return self.pdf_cache.get_stats()
        return {}
    
    def read_file_async(self, file_path: Path) -> Optional[str]:
        """Read file asynchronously."""
        if self.async_io:
            return self.async_io.read_file_sync(file_path)
        # Fallback to synchronous read
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception:
            return None
    
    def write_file_async(self, file_path: Path, content: str) -> bool:
        """Write file asynchronously."""
        if self.async_io:
            return self.async_io.write_file_sync(file_path, content)
        # Fallback to synchronous write
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False
    
    def read_json_async(self, file_path: Path) -> Optional[Dict]:
        """Read JSON file asynchronously."""
        if self.async_io:
            return self.async_io.read_json_async(file_path)
        # Fallback to synchronous read
        try:
            import json
            return json.loads(file_path.read_text(encoding='utf-8'))
        except Exception:
            return None
    
    def write_json_async(self, file_path: Path, data: Dict) -> bool:
        """Write JSON file asynchronously."""
        if self.async_io:
            import json
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return self.async_io.write_file_sync(file_path, content)
        # Fallback to synchronous write
        try:
            import json
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            return True
        except Exception:
            return False
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources."""
        if self.async_io:
            self.async_io.shutdown()
        if self.pdf_cache:
            self.pdf_cache.clear()


def create_advanced_optimizer(config: Dict) -> AdvancedOptimizer:
    """
    Create advanced optimizer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AdvancedOptimizer instance
    """
    opt_config = config.get('advanced_optimization', {})
    
    return AdvancedOptimizer(
        pdf_cache_size_mb=opt_config.get('pdf_cache_size_mb', 10000),
        async_io_workers=opt_config.get('async_io_workers', 8),
        enable_preloading=opt_config.get('enable_preloading', True),
        enable_async_io=opt_config.get('enable_async_io', True)
    )

