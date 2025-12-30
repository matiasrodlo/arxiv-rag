# Testing Documentation

This folder contains test scripts and evaluation reports for the RAG system.

## Test Scripts

### `test_100_papers.py`

Tests the pipeline on 100 random papers and evaluates the quality of JSON output for RAG training.

**Usage:**
```bash
python testing/test_100_papers.py
```

**What it does:**
1. Randomly selects 100 papers from the papers directory
2. Processes each paper through the complete pipeline
3. Evaluates JSON output quality based on:
   - Required fields presence
   - Text structure completeness
   - Chunk quality and metadata
   - Section mapping accuracy
   - Overall RAG suitability

**Output:**
- `test_100_papers_report.json` - Detailed evaluation report
- Console summary with key metrics

**Evaluation Criteria:**
- **Score (0-10)**: Overall quality score
- **Valid**: Papers suitable for RAG (score >= 7.0)
- **Issues**: Critical problems that affect usability
- **Warnings**: Minor issues that may impact quality

## Reports

Test reports are saved as JSON files with detailed evaluation results for each paper.

