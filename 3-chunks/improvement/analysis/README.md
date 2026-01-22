# Chunk Analysis and Fix Scripts

This directory contains scripts for analyzing and fixing chunk quality issues.

## Files

### Analysis Scripts

- **`analyze_all_chunks.py`**
  - Comprehensive analysis of all optimized chunks
  - Reports on quality, consistency, and potential issues
  - Usage: `python chunk-quality/analysis/analyze_all_chunks.py output_improved/`

### Fix Scripts

- **`fix_chunk_issues.py`**
  - Post-processing script to fix remaining chunk issues
  - Filters low-quality chunks (< 0.7 quality score)
  - Splits very long chunks (> 2000 chars)
  - Usage: `python chunk-quality/analysis/fix_chunk_issues.py output_improved output_improved --workers 12`

## Workflow

1. **Initial Optimization**: `chunk-quality/improve_chunks.py`
   - Applies 16 advanced features
   - Creates optimized chunks in `output_improved/`

2. **Analysis**: `chunk-quality/analysis/analyze_all_chunks.py`
   - Analyzes optimized chunks
   - Identifies remaining issues

3. **Fixes**: `chunk-quality/analysis/fix_chunk_issues.py`
   - Fixes low-quality chunks
   - Splits very long chunks
   - Final cleanup

4. **Ready for Embeddings**: Chunks are ready for embedding generation

## Usage

```bash
# Analyze chunks
python chunk-quality/analysis/analyze_all_chunks.py output_improved/

# Fix issues
python chunk-quality/analysis/fix_chunk_issues.py output_improved output_improved \
    --min-quality 0.7 \
    --max-chunk-size 2000 \
    --workers 12
```
