# ArXiv CS RAG System

RAG (Retrieval-Augmented Generation) system for the full arXiv Computer Science corpus (~800,000 papers), designed for [Veritas: A Scientist for Autonomous Research](https://github.com/matiasrodlo/veritas).

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Process papers:
```bash
python run.py
```

3. Query the RAG system:
```bash
python query.py "your query here"
```

## Project Structure

```
arxiv-rag/
├── run.py              # Main pipeline runner
├── query.py            # Query interface
├── config.yaml         # Configuration
├── paper_ids.txt       # Paper ID list
│
├── src/                # Source code
│   ├── core/          # Pipeline orchestration
│   ├── extractors/    # PDF & formula extraction
│   ├── processors/     # Text processing & chunking
│   ├── embeddings/     # Embedding generation
│   ├── storage/        # Vector database
│   └── retrieval/      # Search & retrieval
│
├── papers/             # PDF files
├── output/             # Extracted JSON files
├── data/               # Processed data & vector DB
└── logs/               # Application logs
```

## Features

- Multi-library PDF extraction (PyMuPDF, pdfplumber, pypdf)
- OCR support for scanned PDFs
- Sentence-aware chunking for optimal RAG performance
- Enhanced section detection with fuzzy matching
- Quality scoring and validation
- Structured JSON output with page-level and section metadata

## Scripts

- `scripts/generate_ids.py` - Generate paper ID list from PDFs
- `scripts/analyze.py` - Analyze JSON files for RAG suitability
