# Utilities

Helper scripts for embedding operations.

## Scripts

- **`save_embeddings_to_disk.py`** - Save embeddings to disk as backup
- **`fix_embedding_issues.py`** - Fix embedding-specific issues in chunks

## Usage

```bash
# Save embeddings to disk
python save_embeddings_to_disk.py <embeddings_dir>

# Fix chunk issues
python fix_embedding_issues.py <input_dir> <output_dir> --workers 12
```

## Features

- Disk backup for embeddings
- HTML cleaning
- Text normalization
- Chunk size optimization
