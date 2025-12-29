# ArXiv CS RAG System

Production-ready RAG (Retrieval-Augmented Generation) system for ArXiv Computer Science papers.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate paper IDs (if needed):
```bash
python scripts/generate_paper_ids.py
```

3. Process papers:
```bash
python run_pipeline.py
```

4. Query the system:
```bash
python query.py "your search query"
```

## Project Structure

```
arxiv-rag/
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── run_pipeline.py      # Main pipeline script
├── query.py             # Query interface
├── scripts/             # Utility scripts
│   └── generate_paper_ids.py
└── src/                 # Source code
    ├── pdf_extractor.py
    ├── text_processor.py
    ├── formula_processor.py
    ├── embedder.py
    ├── vector_store.py
    ├── retriever.py
    └── pipeline.py
```

## Configuration

Edit `config.yaml` to customize:
- PDF extraction settings
- Chunking strategy
- Embedding model
- Vector database type
- Retrieval parameters

