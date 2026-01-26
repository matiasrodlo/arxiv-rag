# arXiv RAG Pipeline

A complete pipeline for downloading, processing, and embedding ArXiv Computer Science papers for RAG (Retrieval-Augmented Generation) systems.

## Overview

This repository contains a three-stage pipeline for building a RAG system from ArXiv papers:

1. **Download** - Download papers from ArXiv
2. **Chunk** - Extract text and create semantic chunks
3. **Embed** - Generate embeddings for vector search

## Structure

```
arxiv-rag/
├── 1-downloader/          # Stage 1: Download papers
│   ├── downloader.py       # Main download script
│   ├── deduplicate.py      # Remove duplicate papers
│   └── README.md
│
├── 2-chunks/               # Stage 2: Chunk papers
│   ├── pipeline/           # Initial chunking pipeline
│   │   ├── run.py          # Main entry point
│   │   ├── config.yaml     # Configuration
│   │   └── src/            # Core modules
│   │       ├── core/       # Pipeline orchestrator
│   │       ├── extractors/ # PDF extraction
│   │       └── processors/ # Text chunking
│   │
│   └── improvement/        # Quality improvement
│       ├── improve_chunks.py
│       └── analysis/       # Analysis tools
│
└── 3-embeddings/           # Stage 3: Generate embeddings
    ├── generation/         # Parallel embedding generation
    ├── analysis/           # Embedding quality analysis
    └── utils/              # Utility scripts
```

## Quick Start

### Stage 1: Download Papers

Download ArXiv CS papers with automatic retry and deduplication.

```bash
cd 1-downloader/
python downloader.py
```

**Results:**
- Downloads 105,421 papers across 10 CS categories
- 89.7% success rate (94,520 papers downloaded)
- 70,016 unique papers after deduplication
- Organized as: `{category}/{year_month}/{paper_id}.pdf`

See [1-downloader/README.md](1-downloader/README.md) for details.

### Stage 2: Chunk Papers

Extract text from PDFs and create semantic chunks.

```bash
cd 2-chunks/pipeline/
pip install -r requirements.txt
python run.py --config config.yaml --paper-ids-file paper_ids.txt
```

**Features:**
- Multi-library PDF extraction (PyMuPDF, pdfplumber, pypdf)
- OCR support for scanned documents
- Semantic chunking with section awareness
- Rich metadata (sections, pages, citations)

**Optional: Improve Chunks**

Apply advanced improvements to enhance chunk quality:

```bash
cd 2-chunks/improvement/
pip install -r requirements.txt
python improve_chunks.py ../pipeline/output/ -o output_improved/
```

See [2-chunks/README.md](2-chunks/README.md) for details.

### Stage 3: Generate Embeddings

Generate embeddings for vector search.

```bash
cd 3-embeddings/generation/
python generate_embeddings_parallel.py <chunks_dir> \
    --model all-mpnet-base-v2 \
    --batch-size 1024 \
    --output-dir embeddings/
```

**Features:**
- Parallel embedding generation
- GPU acceleration (MPS/CUDA)
- Batch processing
- Quality analysis tools

See [3-embeddings/README.md](3-embeddings/README.md) for details.

## Pipeline Statistics

### Download Stage
- **Target papers:** 105,421
- **Downloaded:** 94,520 (89.7%)
- **Unique papers:** 70,016
- **Duplicates removed:** 24,489
- **Total size:** 256 GB

### Categories
- cs.AI (17,899), cs.LG (31,093), cs.CV (21,302)
- cs.CL (8,052), cs.CR (7,757), cs.SE (5,815)
- cs.DC (5,372), cs.IR (4,588), cs.MA (1,839), cs.MM (1,704)

## Requirements

### Stage 1 (Downloader)
- Python 3.7+
- Google Cloud SDK (gsutil)
- 300+ GB disk space

### Stage 2 (Chunking)
- Python 3.8+
- See [2-chunks/pipeline/requirements.txt](2-chunks/pipeline/requirements.txt)
- Key libraries: PyMuPDF, sentence-transformers, torch

### Stage 3 (Embeddings)
- Python 3.8+
- See [3-embeddings/generation/README.md](3-embeddings/generation/README.md)
- GPU recommended (MPS for Apple Silicon, CUDA for NVIDIA)

## Workflow

```
1. Download Papers
   └─> PDFs organized by category/year_month

2. Chunk Papers
   └─> JSON files with text, chunks, and metadata

3. Generate Embeddings
   └─> Vector embeddings ready for RAG
```

## Documentation

- [1-downloader/README.md](1-downloader/README.md) - Download system documentation
- [2-chunks/README.md](2-chunks/README.md) - Chunking pipeline documentation
- [3-embeddings/README.md](3-embeddings/README.md) - Embedding generation documentation

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Recent Updates

- Enhanced PDF extraction with improved OCR quality
- Added chunk quality analysis tools
- Optimized embedding generation for better performance
