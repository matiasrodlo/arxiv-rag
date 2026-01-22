# Stage 2: Quality Improvement

Post-processing stage that applies 16 advanced improvements to chunks from Stage 1.

## Usage

```bash
python improve_chunks.py <input_dir> -o <output_dir>
```

## Features

16 advanced improvements including:
- Sentence-aware chunking
- Section-aware chunking (IMRaD boundaries)
- NER and keyword extraction
- Quality filtering
- Context window optimization
- And 11 more...

## Tools

- `improve_chunks.py` - Main improvement script
- `analysis/` - Quality analysis tools
- `scripts/` - Utility scripts

## Output

Enhanced chunks with comprehensive metadata, ready for embedding generation.
