"""
Async I/O operations to reduce CPU wait time.
Allows CPU to continue processing while files are being read/written.
"""

import asyncio
import aiofiles
import json
from pathlib import Path
from typing import Dict, Optional, List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading


class AsyncIOHelper:
    """Helper class for async file operations."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize async I/O helper.
        
        Args:
            max_workers: Maximum number of concurrent I/O operations
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = None
        self._loop_thread = None
    
    def _get_or_create_loop(self):
        """Get or create event loop in a separate thread."""
        if self.loop is None or self.loop.is_closed():
            def run_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            
            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()
            # Wait for loop to be created
            while self.loop is None:
                threading.Event().wait(0.1)
        
        return self.loop
    
    async def read_file_async(self, file_path: Path) -> Optional[str]:
        """Read file asynchronously."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.debug(f"Async read failed for {file_path}: {e}")
            return None
    
    async def write_file_async(self, file_path: Path, content: str) -> bool:
        """Write file asynchronously."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            return True
        except Exception as e:
            logger.debug(f"Async write failed for {file_path}: {e}")
            return False
    
    async def read_json_async(self, file_path: Path) -> Optional[Dict]:
        """Read JSON file asynchronously."""
        try:
            content = await self.read_file_async(file_path)
            if content:
                return json.loads(content)
        except Exception as e:
            logger.debug(f"Async JSON read failed for {file_path}: {e}")
        return None
    
    async def write_json_async(self, file_path: Path, data: Dict) -> bool:
        """Write JSON file asynchronously."""
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return await self.write_file_async(file_path, content)
        except Exception as e:
            logger.debug(f"Async JSON write failed for {file_path}: {e}")
            return False
    
    def read_file_sync(self, file_path: Path) -> Optional[str]:
        """Synchronous wrapper for async read (for compatibility)."""
        loop = self._get_or_create_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.read_file_async(file_path), loop
        )
        return future.result(timeout=30)
    
    def write_file_sync(self, file_path: Path, content: str) -> bool:
        """Synchronous wrapper for async write (for compatibility)."""
        loop = self._get_or_create_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.write_file_async(file_path, content), loop
        )
        return future.result(timeout=30)
    
    async def read_multiple_files(self, file_paths: List[Path]) -> Dict[Path, Optional[str]]:
        """Read multiple files concurrently."""
        tasks = {path: self.read_file_async(path) for path in file_paths}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            path: result if not isinstance(result, Exception) else None
            for path, result in zip(tasks.keys(), results)
        }
    
    async def write_multiple_files(self, file_data: Dict[Path, str]) -> Dict[Path, bool]:
        """Write multiple files concurrently."""
        tasks = {
            path: self.write_file_async(path, content)
            for path, content in file_data.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            path: result if not isinstance(result, Exception) else False
            for path, result in zip(tasks.keys(), results)
        }
    
    def shutdown(self):
        """Shutdown async I/O helper."""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.executor:
            self.executor.shutdown(wait=True)


# Global instance (can be shared across workers)
_async_io_helper = None

def get_async_io_helper(max_workers: int = 4) -> AsyncIOHelper:
    """Get or create global async I/O helper."""
    global _async_io_helper
    if _async_io_helper is None:
        _async_io_helper = AsyncIOHelper(max_workers=max_workers)
    return _async_io_helper

