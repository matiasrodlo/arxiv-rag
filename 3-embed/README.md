# Embeddings Module

Generate, analyze, and optimize embeddings for RAG systems.

## Structure

```
embeddings/
├── generation/          # Embedding generation
│   └── generate_embeddings_parallel.py
│
├── analysis/           # Quality analysis
│   ├── analyze_for_embeddings.py
│   └── final_embedding_analysis.py
│
├── utils/              # Utilities
│   ├── save_embeddings_to_disk.py
│   └── fix_embedding_issues.py
│
└── README.md           # This file
```

## Quick Start

### Generate Embeddings

```bash
cd embeddings/generation
python generate_embeddings_parallel.py <chunks_dir> \
    --model all-mpnet-base-v2 \
    --chroma-db ./chroma_db \
    --min-quality 0.9 \
    --batch-size 200
```

### Analyze Chunks

```bash
cd embeddings/analysis
python analyze_for_embeddings.py <chunks_dir>
```

### Fix Issues

```bash
cd embeddings/utils
python fix_embedding_issues.py <input_dir> <output_dir>
```

## Features

- **Parallel Processing**: Optimized for M4 Pro Max (128GB RAM)
- **Stable Generation**: Comprehensive error handling and recovery
- **Quality Analysis**: Token counts, text quality, context validation
- **Issue Fixing**: Automatic cleanup and optimization
- **ChromaDB Integration**: Persistent vector storage

## Output

Embeddings are stored in:
- **ChromaDB**: `./chroma_db/` (default)
- **Collection**: `scientific_papers` (default)
- **Disk Backup**: `.npy` files with metadata
