#!/usr/bin/env python3
"""
Generate embeddings for all optimized chunks with stable parallel processing.
Optimized for M4 Pro Max with 128GB RAM - utilizes 80%+ CPU with stability.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import gc
import time
import traceback
import sys

# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not available. Install with: pip install chromadb")

# Worker-level model cache (each process gets its own)
_worker_model = None
_worker_model_name = None

def init_worker_model(model_name: str):
    """Initialize model in worker process (called once per worker)."""
    global _worker_model, _worker_model_name
    
    if _worker_model is None or _worker_model_name != model_name:
        try:
            print(f"[Worker {mp.current_process().name}] Loading model: {model_name}")
            _worker_model = SentenceTransformer(model_name)
            
            # Try to use MPS if available (but only for single worker to avoid conflicts)
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Only use MPS if we have few workers (MPS doesn't handle multiple processes well)
                    # For now, use CPU for stability
                    pass  # _worker_model = _worker_model.to('mps')
            except:
                pass
            
            _worker_model_name = model_name
            print(f"[Worker {mp.current_process().name}] Model loaded successfully")
        except Exception as e:
            print(f"[Worker {mp.current_process().name}] Error loading model: {e}")
            traceback.print_exc()
            _worker_model = None
            raise

def get_worker_model(model_name: str):
    """Get model for current worker (initializes if needed)."""
    global _worker_model, _worker_model_name
    
    if _worker_model is None or _worker_model_name != model_name:
        init_worker_model(model_name)
    
    return _worker_model

def load_chunks_from_file(file_path: Path, min_quality: float = 0.8) -> List[Dict]:
    """Load chunks from a single file."""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_chunks = data.get('chunks', [])
        for chunk in file_chunks:
            quality = chunk.get('metadata', {}).get('quality_score', 0)
            if quality >= min_quality:
                chunks.append(chunk)
    except Exception as e:
        # Silent failure for individual files
        pass
    
    return chunks

def load_chunks_parallel(output_dir: Path, 
                        min_quality: float = 0.8,
                        num_workers: int = None) -> List[Dict]:
    """Load all chunks from optimized files using parallel processing."""
    all_files = list(output_dir.rglob('*.json'))
    all_files = [f for f in all_files if not f.name.startswith('._')]
    
    if num_workers is None:
        # Use 80% of CPU cores for loading (aggressive)
        num_workers = max(1, int(mp.cpu_count() * 0.8))
    
    print(f"Loading chunks from {len(all_files):,} files using {num_workers} workers...")
    
    # Parallel loading
    results = []
    try:
        with mp.Pool(processes=num_workers) as pool:
            worker_func = partial(load_chunks_from_file, min_quality=min_quality)
            
            for result in tqdm(pool.imap_unordered(worker_func, all_files), 
                              total=len(all_files), desc="Loading files"):
                if result:
                    results.extend(result)
                # Periodic garbage collection
                if len(results) % 10000 == 0:
                    gc.collect()
    except KeyboardInterrupt:
        print("\n⚠️  Loading interrupted by user")
        raise
    except Exception as e:
        print(f"Error during loading: {e}")
        traceback.print_exc()
        raise
    
    print(f"Loaded {len(results):,} high-quality chunks (quality >= {min_quality})")
    return results

def generate_embeddings_worker(args: Tuple[List[str], str, int, int]) -> Tuple[int, List[np.ndarray], Optional[str]]:
    """Worker function for parallel embedding generation with robust error handling."""
    texts_batch, model_name, batch_size, batch_idx = args
    
    try:
        model = get_worker_model(model_name)
        if model is None:
            return (batch_idx, [], f"Model not loaded in worker {mp.current_process().name}")
        
        embeddings = []
        
        # Process in sub-batches with error handling
        for i in range(0, len(texts_batch), batch_size):
            batch = texts_batch[i:i+batch_size]
            try:
                batch_embeddings = model.encode(
                    batch,
                    batch_size=min(batch_size, len(batch)),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False  # Keep as numpy for stability
                )
                embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails, try individual items
                print(f"[Worker {mp.current_process().name}] Batch {i//batch_size} failed, trying individual items")
                for text in batch:
                    try:
                        emb = model.encode(
                            [text],
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        embeddings.extend(emb)
                    except:
                        # Use zero vector as fallback
                        embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
        
        return (batch_idx, embeddings, None)
    except Exception as e:
        error_msg = f"Error in worker {mp.current_process().name}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        # Return empty embeddings with error message
        return (batch_idx, [], error_msg)

def generate_embeddings_parallel(chunks: List[Dict],
                                model_name: str = 'all-mpnet-base-v2',
                                batch_size: int = 200,
                                num_workers: int = None,
                                chunks_per_worker: int = 500) -> List[np.ndarray]:
    """Generate embeddings using true parallel processing with stability improvements."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not available")
    
    if num_workers is None:
        # Use 75% of CPU cores for embedding generation
        num_workers = max(1, int(mp.cpu_count() * 0.75))
    
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts):,} chunks using {num_workers} workers...")
    print(f"Model: {model_name}, Batch size: {batch_size}, Chunks per worker: {chunks_per_worker}")
    
    # Split chunks into batches for workers
    text_batches = []
    for i in range(0, len(texts), chunks_per_worker):
        text_batches.append((texts[i:i+chunks_per_worker], model_name, batch_size, len(text_batches)))
    
    # Initialize workers with model
    print(f"Initializing {num_workers} workers with model...")
    try:
        with mp.Pool(processes=num_workers, initializer=init_worker_model, initargs=(model_name,)) as pool:
            # Generate embeddings in parallel with order preservation
            embeddings_dict = {}
            errors = []
            
            print(f"Processing {len(text_batches)} batches across {num_workers} workers...")
            worker_func = generate_embeddings_worker
            
            for result in tqdm(
                pool.imap_unordered(worker_func, text_batches),
                total=len(text_batches),
                desc="Generating embeddings"
            ):
                batch_idx, batch_embeddings, error = result
                
                if error:
                    errors.append(f"Batch {batch_idx}: {error}")
                
                if batch_embeddings:
                    embeddings_dict[batch_idx] = batch_embeddings
                else:
                    print(f"Warning: Batch {batch_idx} produced no embeddings")
            
            # Report errors
            if errors:
                print(f"\n⚠️  {len(errors)} batches had errors:")
                for err in errors[:10]:  # Show first 10
                    print(f"  - {err}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more")
            
            # Reorder embeddings to match original chunk order
            embeddings = []
            missing_batches = []
            for i in range(len(text_batches)):
                if i in embeddings_dict:
                    embeddings.extend(embeddings_dict[i])
                else:
                    missing_batches.append(i)
                    # Create zero vectors as fallback
                    try:
                        # Get embedding dimension from first successful batch
                        if embeddings:
                            dim = len(embeddings[0])
                        else:
                            # Try to get from model
                            model = SentenceTransformer(model_name)
                            dim = model.get_sentence_embedding_dimension()
                        embeddings.extend([np.zeros(dim)] * len(text_batches[i]))
                    except:
                        print(f"Error creating fallback for batch {i}")
            
            if missing_batches:
                print(f"⚠️  {len(missing_batches)} batches missing, used zero vectors as fallback")
            
    except KeyboardInterrupt:
        print("\n⚠️  Embedding generation interrupted by user")
        raise
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        traceback.print_exc()
        raise
    
    print(f"Generated {len(embeddings):,} embeddings")
    
    # Validate embeddings
    if len(embeddings) != len(texts):
        print(f"⚠️  Warning: Embedding count ({len(embeddings)}) doesn't match text count ({len(texts)})")
        # Pad or truncate to match
        if len(embeddings) < len(texts):
            dim = len(embeddings[0]) if embeddings else 768
            embeddings.extend([np.zeros(dim)] * (len(texts) - len(embeddings)))
        else:
            embeddings = embeddings[:len(texts)]
    
    return embeddings

def save_embeddings_to_disk_fallback(chunks: List[Dict],
                                     embeddings: List,
                                     output_dir: str = "./embeddings_saved"):
    """Save embeddings to disk as fallback when ChromaDB fails."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n⚠️  ChromaDB storage failed. Saving embeddings to disk: {output_dir}")
    
    # Convert to numpy array
    if isinstance(embeddings[0], list):
        embeddings_array = np.array(embeddings)
    else:
        embeddings_array = np.array([e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings])
    
    # Save embeddings
    embeddings_file = output_path / "embeddings.npy"
    np.save(embeddings_file, embeddings_array)
    print(f"✅ Saved embeddings to {embeddings_file}")
    
    # Save metadata
    metadata = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get('metadata', {})
        metadata.append({
            'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
            'paper_id': meta.get('paper_id'),
            'section': meta.get('section'),
            'quality_score': meta.get('quality_score'),
            'text': chunk['text'],
            'text_length': len(chunk['text'])
        })
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved metadata to {metadata_file}")
    
    # Save index
    index = {
        'total_embeddings': len(embeddings),
        'embedding_dimension': embeddings_array.shape[1],
        'total_size_gb': embeddings_array.nbytes / 1024 / 1024 / 1024
    }
    
    index_file = output_path / "index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ Saved {len(embeddings):,} embeddings ({embeddings_array.shape[1]} dimensions)")
    print(f"   Total size: ~{embeddings_array.nbytes / 1024 / 1024 / 1024:.2f} GB")

def store_in_chroma_batched(chunks: List[Dict],
                            embeddings: List,
                            db_path: str = "./chroma_db",
                            collection_name: str = "scientific_papers",
                            batch_size: int = 2000):
    """Store embeddings in ChromaDB with optimized batching and robust error handling."""
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not available")
    
    print(f"Storing embeddings in ChromaDB: {db_path}")
    
    # Validate inputs
    if len(chunks) != len(embeddings):
        print(f"⚠️  Warning: Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})")
        min_len = min(len(chunks), len(embeddings))
        chunks = chunks[:min_len]
        embeddings = embeddings[:min_len]
        print(f"Using first {min_len} chunks and embeddings")
    
    # Fix permissions first
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        print(f"Fixing permissions for {db_path}...")
        try:
            import os
            import stat
            # Make directory writable
            os.chmod(db_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
            # Make all files in directory writable
            for root, dirs, files in os.walk(db_path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
                for f in files:
                    os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            print("✅ Permissions fixed")
        except Exception as e:
            print(f"⚠️  Could not fix permissions: {e}")
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # Create collection if it doesn't exist
        try:
            collection = client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Scientific paper chunks for RAG"}
            )
            print(f"Created new collection: {collection_name}")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        traceback.print_exc()
        # Fallback to disk
        save_embeddings_to_disk_fallback(chunks, embeddings)
        raise
    
    print(f"Adding {len(chunks):,} embeddings in batches of {batch_size}...")
    
    # Store in large batches (sequential for ChromaDB stability, but large batches are fast)
    total_stored = 0
    failed_batches = 0
    
    try:
        for i in tqdm(range(0, len(chunks), batch_size), desc="Storing embeddings"):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            ids = []
            embeds = []
            docs = []
            metas = []
            
            for j, chunk in enumerate(batch_chunks):
                embedding = batch_embeddings[j]
                metadata = chunk.get('metadata', {})
                
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                ids.append(chunk.get('chunk_id', f"chunk_{i+j}"))
                embeds.append(embedding)
                docs.append(chunk['text'])
                metas.append({
                    'paper_id': str(metadata.get('paper_id', 'unknown')),
                    'section': str(metadata.get('section', 'unknown')),
                    'quality_score': float(metadata.get('quality_score', 0.0)),
                    'chunk_index': str(metadata.get('chunk_index', i+j)),
                })
            
            # Batch add with retry logic
            max_retries = 3
            stored = False
            for retry in range(max_retries):
                try:
                    collection.add(
                        ids=ids,
                        embeddings=embeds,
                        documents=docs,
                        metadatas=metas
                    )
                    total_stored += len(batch_chunks)
                    stored = True
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"Retry {retry + 1}/{max_retries} for batch {i//batch_size}")
                        time.sleep(0.5)  # Brief pause before retry
                    else:
                        print(f"Error storing batch {i//batch_size}: {e}")
                        failed_batches += 1
                        # Try individual adds as fallback
                        for k in range(len(batch_chunks)):
                            try:
                                collection.add(
                                    ids=[ids[k]],
                                    embeddings=[embeds[k]],
                                    documents=[docs[k]],
                                    metadatas=[metas[k]]
                                )
                                total_stored += 1
                            except:
                                pass
            
            # Periodic garbage collection
            if i % (batch_size * 10) == 0:
                gc.collect()
    except KeyboardInterrupt:
        print("\n⚠️  Storage interrupted by user")
        raise
    except Exception as e:
        print(f"Error during storage: {e}")
        traceback.print_exc()
        raise
    
        if failed_batches > 0:
            print(f"⚠️  {failed_batches} batches failed, but individual items were stored")
    
    if total_stored == 0:
        print(f"\n❌ Failed to store any embeddings in ChromaDB")
        print(f"⚠️  Saving embeddings to disk as fallback...")
        save_embeddings_to_disk_fallback(chunks, embeddings)
        print(f"\n⚠️  You can load these embeddings later and retry ChromaDB storage")
    else:
        print(f"✅ Stored {total_stored:,} embeddings in ChromaDB")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings with stable parallel processing')
    parser.add_argument('input_dir', type=str, help='Input directory (output_improved)')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                       choices=['all-mpnet-base-v2', 'all-MiniLM-L6-v2'],
                       help='Embedding model to use')
    parser.add_argument('--min-quality', type=float, default=0.9,
                       help='Minimum quality score (default: 0.9)')
    parser.add_argument('--batch-size', type=int, default=200,
                       help='Batch size for embedding generation (default: 200, stable)')
    parser.add_argument('--chroma-db', type=str, default='./chroma_db',
                       help='Path to ChromaDB (default: ./chroma_db)')
    parser.add_argument('--collection-name', type=str, default='scientific_papers',
                       help='ChromaDB collection name')
    parser.add_argument('--load-workers', type=int, default=None,
                       help='Number of workers for loading (default: 80% of CPU cores)')
    parser.add_argument('--embedding-workers', type=int, default=None,
                       help='Number of workers for embedding (default: 75% of CPU cores)')
    parser.add_argument('--store-batch-size', type=int, default=2000,
                       help='Batch size for storing in ChromaDB (default: 2000, sequential for stability)')
    parser.add_argument('--chunks-per-worker', type=int, default=500,
                       help='Chunks per embedding worker (default: 500)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found")
        return 1
    
    cpu_count = mp.cpu_count()
    print("=" * 80)
    print("STABLE PARALLEL EMBEDDING GENERATION - M4 PRO MAX")
    print("=" * 80)
    print()
    print(f"System: {cpu_count} CPU cores, 128GB RAM")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Min quality: {args.min_quality}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Load workers: {args.load_workers or int(cpu_count * 0.8)}")
    print(f"  Embedding workers: {args.embedding_workers or int(cpu_count * 0.75)}")
    print(f"  Store batch size: {args.store_batch_size}")
    print(f"  ChromaDB: {args.chroma_db}")
    print()
    
    try:
        start_time = time.time()
        
        # Load chunks in parallel
        chunks = load_chunks_parallel(
            input_dir,
            min_quality=args.min_quality,
            num_workers=args.load_workers
        )
        
        if not chunks:
            print("No chunks found!")
            return 1
        
        load_time = time.time() - start_time
        print(f"Loading completed in {load_time:.1f} seconds")
        print()
        
        # Generate embeddings in parallel
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Error: sentence-transformers not available")
            print("Install with: pip install sentence-transformers")
            return 1
        
        embedding_start = time.time()
        embeddings = generate_embeddings_parallel(
            chunks,
            model_name=args.model,
            batch_size=args.batch_size,
            num_workers=args.embedding_workers,
            chunks_per_worker=args.chunks_per_worker
        )
        embedding_time = time.time() - embedding_start
        print(f"Embedding generation completed in {embedding_time:.1f} seconds ({embedding_time/60:.1f} min)")
        print()
        
        # Save embeddings to disk FIRST as backup
        print("Saving embeddings to disk as backup...")
        save_embeddings_to_disk_fallback(chunks, embeddings, output_dir="./embeddings_saved")
        print()
        
        # Store in ChromaDB with parallel processing
        if not CHROMA_AVAILABLE:
            print("Error: chromadb not available")
            print("Install with: pip install chromadb")
            return 1
        
        store_start = time.time()
        try:
            store_in_chroma_batched(
                chunks,
                embeddings,
                db_path=args.chroma_db,
                collection_name=args.collection_name,
                batch_size=args.store_batch_size
            )
            store_time = time.time() - store_start
            print(f"Storage completed in {store_time:.1f} seconds")
        except Exception as e:
            store_time = time.time() - store_start
            print(f"⚠️  ChromaDB storage failed after {store_time:.1f} seconds: {e}")
            print(f"✅ Embeddings are safely saved to ./embeddings_saved/")
            print(f"You can load them later and retry ChromaDB storage")
        print()
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 80)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 80)
        print(f"Chunks processed: {len(chunks):,}")
        print(f"Embeddings generated: {len(embeddings):,}")
        if embeddings:
            print(f"Embedding dimensions: {len(embeddings[0])}")
        print(f"Stored in: {args.chroma_db}")
        print()
        print(f"Timing:")
        print(f"  Loading: {load_time:.1f}s")
        print(f"  Embedding: {embedding_time:.1f}s ({embedding_time/60:.1f} min)")
        print(f"  Storage: {store_time:.1f}s")
        print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    # Set multiprocessing start method for macOS
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    sys.exit(main())
