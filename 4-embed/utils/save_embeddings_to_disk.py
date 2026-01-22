#!/usr/bin/env python3
"""
Save embeddings to disk as numpy arrays and JSON metadata.
Useful when ChromaDB has permission issues.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm
import pickle

def save_embeddings_disk(chunks: List[Dict], 
                        embeddings: List,
                        output_dir: str = "./embeddings_saved",
                        batch_size: int = 10000):
    """Save embeddings to disk in batches."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(embeddings):,} embeddings to {output_dir}...")
    
    # Convert embeddings to numpy array
    if isinstance(embeddings[0], list):
        embeddings_array = np.array(embeddings)
    else:
        embeddings_array = np.array([e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings])
    
    # Save embeddings in batches
    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Saving batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings))
        
        batch_embeddings = embeddings_array[start_idx:end_idx]
        batch_chunks = chunks[start_idx:end_idx]
        
        # Save embeddings
        embeddings_file = output_path / f"embeddings_batch_{i:04d}.npy"
        np.save(embeddings_file, batch_embeddings)
        
        # Save metadata
        metadata = []
        for j, chunk in enumerate(batch_chunks):
            meta = chunk.get('metadata', {})
            metadata.append({
                'chunk_id': chunk.get('chunk_id', f'chunk_{start_idx + j}'),
                'paper_id': meta.get('paper_id'),
                'section': meta.get('section'),
                'quality_score': meta.get('quality_score'),
                'text': chunk['text'],
                'text_length': len(chunk['text']),
                'batch_index': i,
                'batch_position': j
            })
        
        metadata_file = output_path / f"metadata_batch_{i:04d}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save index file
    index = {
        'total_embeddings': len(embeddings),
        'embedding_dimension': embeddings_array.shape[1],
        'num_batches': num_batches,
        'batch_size': batch_size,
        'chunks': [chunk.get('chunk_id') for chunk in chunks]
    }
    
    index_file = output_path / "index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ Saved {len(embeddings):,} embeddings to {output_dir}")
    print(f"   - {num_batches} batch files")
    print(f"   - Embedding dimension: {embeddings_array.shape[1]}")
    print(f"   - Total size: ~{embeddings_array.nbytes / 1024 / 1024 / 1024:.2f} GB")

def load_embeddings_from_disk(input_dir: str, batch_index: int = None):
    """Load embeddings from disk."""
    input_path = Path(input_dir)
    
    if batch_index is not None:
        # Load specific batch
        embeddings_file = input_path / f"embeddings_batch_{batch_index:04d}.npy"
        metadata_file = input_path / f"metadata_batch_{batch_index:04d}.json"
        
        embeddings = np.load(embeddings_file)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return embeddings, metadata
    else:
        # Load all batches
        index_file = input_path / "index.json"
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        all_embeddings = []
        all_metadata = []
        
        for i in range(index['num_batches']):
            embeddings_file = input_path / f"embeddings_batch_{i:04d}.npy"
            metadata_file = input_path / f"metadata_batch_{i:04d}.json"
            
            batch_embeddings = np.load(embeddings_file)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                batch_metadata = json.load(f)
            
            all_embeddings.append(batch_embeddings)
            all_metadata.extend(batch_metadata)
        
        embeddings = np.vstack(all_embeddings)
        return embeddings, all_metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save embeddings to disk')
    parser.add_argument('--chunks-file', type=str, required=True,
                       help='JSON file with chunks data')
    parser.add_argument('--embeddings-file', type=str, required=True,
                       help='Pickle or numpy file with embeddings')
    parser.add_argument('--output-dir', type=str, default='./embeddings_saved',
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size for saving')
    
    args = parser.parse_args()
    
    # This script would need to be called with the actual chunks and embeddings
    # For now, it's a utility script
    print("Use this script to save embeddings when ChromaDB fails")
    print("You'll need to pass chunks and embeddings from the main script")
