# RAG Pipeline System

A two-stage RAG (Retrieval Augmented Generation) pipeline for processing scientific papers and generating high-quality embeddings.

## Structure

```
chunks/
├── pipeline/          # Stage 1: Initial Processing
│   ├── run.py         # Main entry point
│   ├── config.yaml    # Configuration
│   ├── requirements.txt
│   └── src/           # Core pipeline modules
│       ├── core/      # Pipeline orchestrator & workers
│       ├── extractors/ # PDF extraction & formula processing
│       └── processors/ # Text processing & chunking
│
├── improvement/       # Stage 2: Quality Improvement
│   ├── improve_chunks.py  # 16 advanced improvements
│   ├── requirements.txt
│   ├── analysis/      # Quality analysis tools
│   └── scripts/       # Utility scripts
│
└── docs/              # Documentation
    ├── ANALYSIS_REPORT.md
    ├── FOLDER_ANALYSIS.md
    └── REDUNDANCY_ANALYSIS.md
```

## Workflow

### Stage 1: Initial Processing
Processes PDFs, extracts text, and creates initial chunks.

```bash
cd pipeline/
python run.py --config config.yaml
```

### Stage 2: Quality Improvement
Applies 16 advanced improvements to enhance chunk quality.

```bash
cd improvement/
python improve_chunks.py <input_dir> -o <output_dir>
```

## Quick Start

### Stage 1: Process PDFs
```bash
cd chunks/pipeline
pip install -r requirements.txt
python run.py --config config.yaml
```

### Stage 2: Improve Chunks
```bash
cd chunks/improvement
pip install -r requirements.txt
python improve_chunks.py ../pipeline/output/ -o output_improved/
```

## Features

### Stage 1 Pipeline
- Multi-library PDF extraction (PyMuPDF, pdfplumber, pypdf)
- OCR support for scanned documents
- Intelligent text processing
- Section-aware chunking
- Vector database indexing

### Stage 2 Improvement
- 16 advanced chunk improvements
- Sentence-aware chunking
- NER and keyword extraction
- Quality filtering
- Context window optimization

## Documentation

See `docs/` for detailed analysis and reports.
