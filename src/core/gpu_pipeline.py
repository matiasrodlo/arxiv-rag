"""
GPU Pipeline Manager.
Keeps GPU busy by pipelining operations and prefetching data.
"""

import torch
from typing import List, Optional, Callable
from collections import deque
from loguru import logger
import threading
import queue


class GPUPipeline:
    """
    Pipeline GPU operations to maximize utilization.
    Pre-fetches next batch while processing current batch.
    """
    
    def __init__(self, 
                 model,
                 device: str = "mps",
                 batch_size: int = 512,
                 pipeline_depth: int = 2):
        """
        Initialize GPU pipeline.
        
        Args:
            model: Model to use for inference
            device: Device (mps, cuda, cpu)
            batch_size: Batch size for processing
            pipeline_depth: Number of batches to pipeline
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.pipeline_depth = pipeline_depth
        
        # Pipeline queues
        self.input_queue = queue.Queue(maxsize=pipeline_depth)
        self.output_queue = queue.Queue(maxsize=pipeline_depth)
        
        # Worker thread
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start pipeline worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.debug("GPU pipeline started")
    
    def stop(self):
        """Stop pipeline worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.debug("GPU pipeline stopped")
    
    def _worker(self):
        """Worker thread that processes batches."""
        while self.running:
            try:
                # Get batch from input queue
                batch_data = self.input_queue.get(timeout=1)
                if batch_data is None:  # Poison pill
                    break
                
                texts, callback = batch_data
                
                # Process on GPU
                with torch.inference_mode():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                
                # Put result in output queue
                self.output_queue.put((embeddings, callback))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU pipeline worker error: {e}")
                continue
    
    def process_async(self, texts: List[str], callback: Optional[Callable] = None):
        """
        Queue texts for async processing.
        
        Args:
            texts: Texts to process
            callback: Optional callback function
        """
        if not self.running:
            self.start()
        
        try:
            self.input_queue.put((texts, callback), timeout=5)
        except queue.Full:
            logger.warning("GPU pipeline queue full, processing synchronously")
            # Fallback to synchronous processing
            with torch.inference_mode():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            if callback:
                callback(embeddings)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[List]:
        """
        Get result from pipeline.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Embeddings or None if timeout
        """
        try:
            result, callback = self.output_queue.get(timeout=timeout)
            if callback:
                callback(result)
            return result
        except queue.Empty:
            return None


def create_gpu_pipeline(model, 
                       device: str = "mps",
                       batch_size: int = 512,
                       pipeline_depth: int = 2) -> GPUPipeline:
    """
    Create GPU pipeline.
    
    Args:
        model: Model to use
        device: Device
        batch_size: Batch size
        pipeline_depth: Pipeline depth
        
    Returns:
        GPUPipeline instance
    """
    return GPUPipeline(
        model=model,
        device=device,
        batch_size=batch_size,
        pipeline_depth=pipeline_depth
    )

