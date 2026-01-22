# Embedding Generation

Main script for generating embeddings from optimized chunks.

## Usage

```bash
python generate_embeddings_parallel.py <chunks_dir> \
    --model all-mpnet-base-v2 \
    --chroma-db ./chroma_db \
    --min-quality 0.9 \
    --batch-size 200 \
    --store-batch-size 2000
```

## Features

- Parallel processing (optimized for M4 Pro Max)
- Stable error handling
- ChromaDB integration
- Disk backup fallback
- Progress tracking

## Output

- ChromaDB collection: `scientific_papers`
- Disk backup: `.npy` files with metadata
