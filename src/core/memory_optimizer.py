"""
Memory-based optimizations to unlock CPU bottlenecks.
Uses idle RAM to speed up processing through caching and batch operations.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from loguru import logger
import multiprocessing as mp


class MemoryOptimizer:
    """Optimize memory usage to reduce CPU bottlenecks."""
    
    def __init__(self, 
                 use_ram_disk: bool = True,
                 ram_disk_size_gb: int = 20,
                 ram_disk_path: Optional[str] = None,
                 enable_model_caching: bool = True,
                 max_workers: int = 12):
        """
        Initialize memory optimizer.
        
        Args:
            use_ram_disk: Use RAM disk for temporary files and cache
            ram_disk_size_gb: Size of RAM disk in GB
            ram_disk_path: Custom path for RAM disk (auto-created if None)
            enable_model_caching: Cache models in shared memory
            max_workers: Number of workers to optimize for
        """
        self.use_ram_disk = use_ram_disk
        self.ram_disk_size_gb = ram_disk_size_gb
        self.ram_disk_path = ram_disk_path
        self.enable_model_caching = enable_model_caching
        self.max_workers = max_workers
        
        self.ram_disk_mount_point = None
        self.original_cache_dir = None
        
    def setup_ram_disk(self) -> Optional[Path]:
        """
        Set up RAM disk for faster I/O operations.
        
        Returns:
            Path to RAM disk mount point, or None if setup failed
        """
        if not self.use_ram_disk:
            return None
            
        try:
            # Check if we're on macOS (which uses diskutil)
            import platform
            if platform.system() != 'Darwin':
                logger.warning("RAM disk setup currently optimized for macOS. Skipping.")
                return None
            
            # Use provided path or create temp directory
            if self.ram_disk_path:
                mount_point = Path(self.ram_disk_path)
            else:
                # Use /tmp/arxiv_rag_ramdisk (faster than regular disk on macOS)
                mount_point = Path("/tmp/arxiv_rag_ramdisk")
            
            # Create mount point if it doesn't exist
            mount_point.mkdir(parents=True, exist_ok=True)
            
            # Try to create RAM disk using diskutil (macOS)
            # Note: This requires admin privileges, so we'll use a simpler approach
            # Instead of creating a system RAM disk, we'll use /tmp which is often on fast storage
            # For true RAM disk, user can create manually: hdiutil attach -nomount ram://20971520
            
            # Check if user has created a RAM disk manually
            ram_disk_name = "ArxivRAG_RAMDisk"
            ram_disk_vol = Path(f'/Volumes/{ram_disk_name}')
            
            if ram_disk_vol.exists():
                # User created RAM disk manually, use it
                cache_dir = ram_disk_vol / 'cache'
                cache_dir.mkdir(exist_ok=True)
                self.ram_disk_mount_point = ram_disk_vol
                logger.info(f"Using existing RAM disk at: {ram_disk_vol}")
                return ram_disk_vol
            
            # Fallback: Use /tmp (which is often on faster storage)
            logger.info(f"Using /tmp for fast cache storage: {mount_point}")
            mount_point.mkdir(parents=True, exist_ok=True)
            return mount_point
            
        except Exception as e:
            logger.warning(f"Failed to setup RAM disk: {e}. Using regular disk cache.")
            return None
    
    def get_cache_directory(self, base_cache_dir: Path) -> Path:
        """
        Get optimized cache directory (RAM disk if available, else regular).
        
        Args:
            base_cache_dir: Original cache directory path
            
        Returns:
            Optimized cache directory path
        """
        if self.ram_disk_mount_point:
            cache_dir = self.ram_disk_mount_point / 'cache' / 'pdf_extraction'
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using RAM disk cache: {cache_dir}")
            return cache_dir
        else:
            # Use /tmp which is often faster than regular disk
            tmp_cache = Path("/tmp/arxiv_rag_cache")
            tmp_cache.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using /tmp cache: {tmp_cache}")
            return tmp_cache
    
    def optimize_worker_memory(self, config: dict) -> dict:
        """
        Optimize configuration for better memory usage.
        
        Args:
            config: Original configuration dictionary
            
        Returns:
            Optimized configuration
        """
        optimized_config = config.copy()
        
        # Increase batch sizes to use more RAM (reduces overhead)
        if 'embeddings' in optimized_config:
            # Larger batch size = more RAM, but fewer model calls = less CPU
            original_batch = optimized_config['embeddings'].get('batch_size', 32)
            # Increase batch size if we have RAM available
            optimized_config['embeddings']['batch_size'] = min(256, original_batch * 2)
            logger.info(f"Optimized embedding batch size: {original_batch} -> {optimized_config['embeddings']['batch_size']}")
        
        # Enable caching for PDF extraction (uses RAM but saves CPU)
        if 'pdf_extraction' not in optimized_config:
            optimized_config['pdf_extraction'] = {}
        optimized_config['pdf_extraction']['enable_caching'] = True
        
        # Optimize processing batch size
        if 'processing' in optimized_config:
            # Larger batch size = more papers in memory = better CPU utilization
            original_batch = optimized_config['processing'].get('batch_size', 1000)
            # Don't increase too much, but ensure it's optimal
            optimized_config['processing']['batch_size'] = max(500, min(2000, original_batch))
        
        return optimized_config
    
    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            
            return {
                'process_memory_mb': memory_info.rss / (1024 * 1024),
                'process_memory_percent': process.memory_percent(),
                'system_total_gb': system_memory.total / (1024 ** 3),
                'system_available_gb': system_memory.available / (1024 ** 3),
                'system_used_percent': system_memory.percent,
                'system_free_percent': 100 - system_memory.percent
            }
        except ImportError:
            logger.warning("psutil not available for memory stats")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return {}
    
    def recommend_worker_count(self) -> int:
        """
        Recommend optimal worker count based on available memory.
        
        Returns:
            Recommended number of workers
        """
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            available_gb = system_memory.available / (1024 ** 3)
            
            # Estimate memory per worker (models + data)
            # Each worker needs: ~2GB for models + ~500MB for processing = ~2.5GB
            memory_per_worker_gb = 2.5
            
            # Calculate max workers based on memory (leave 20% free)
            max_workers_by_memory = int((available_gb * 0.8) / memory_per_worker_gb)
            
            # Don't exceed CPU count
            cpu_count = mp.cpu_count()
            
            # Recommended: min of memory-based and CPU-based, but at least 8
            recommended = max(8, min(max_workers_by_memory, cpu_count, self.max_workers))
            
            logger.info(f"Memory-based worker recommendation: {recommended} workers")
            logger.info(f"  - Available RAM: {available_gb:.1f}GB")
            logger.info(f"  - Memory per worker: ~{memory_per_worker_gb}GB")
            logger.info(f"  - CPU cores: {cpu_count}")
            
            return recommended
            
        except Exception as e:
            logger.warning(f"Could not calculate worker recommendation: {e}")
            return self.max_workers

