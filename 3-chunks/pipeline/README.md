# Stage 1: Initial Processing Pipeline

Main RAG pipeline for processing ArXiv papers and building the vector database.

## Usage

```bash
python run.py --config config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- PDF extraction settings
- Text processing parameters
- Chunking configuration
- Vector database paths

## Structure

- `run.py` - Main entry point
- `src/core/` - Pipeline orchestrator and workers
- `src/extractors/` - PDF extraction and formula processing
- `src/processors/` - Text processing and chunking

## Output

Generates JSON chunk files ready for Stage 2 improvement.
