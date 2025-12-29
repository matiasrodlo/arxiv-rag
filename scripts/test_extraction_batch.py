#!/usr/bin/env python3
"""
Test extraction and text processing on a batch of papers.
Tests the parts that don't require heavy dependencies.
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_extractor import PDFExtractor, load_metadata
from src.text_processor import TextProcessor
from src.formula_processor import FormulaProcessor


def test_paper(paper_id: str, pdf_dir: Path, metadata_dir: Path,
               extractor: PDFExtractor, processor: TextProcessor,
               formula_processor: FormulaProcessor) -> dict:
    """Test a single paper."""
    pdf_path = pdf_dir / f"{paper_id}.pdf"
    
    result = {
        'paper_id': paper_id,
        'success': False,
        'extraction_time': 0,
        'text_length': 0,
        'cleaned_length': 0,
        'num_pages': 0,
        'extraction_method': None,
        'formulas_detected': 0,
        'greek_letters': 0,
        'has_abstract': False,
        'has_introduction': False,
        'has_references': False,
        'error': None
    }
    
    if not pdf_path.exists():
        result['error'] = 'PDF not found'
        return result
    
    try:
        start_time = time.time()
        
        # Extract
        extraction_result = extractor.extract(str(pdf_path))
        result['extraction_time'] = time.time() - start_time
        
        if not extraction_result['success']:
            result['error'] = extraction_result.get('error', 'Extraction failed')
            return result
        
        result['success'] = True
        result['extraction_method'] = extraction_result['method_used']
        result['num_pages'] = len(extraction_result.get('pages', []))
        result['text_length'] = len(extraction_result.get('text', ''))
        
        # Clean text
        cleaned_text = processor.clean(extraction_result['text'])
        result['cleaned_length'] = len(cleaned_text)
        
        # Check structure
        text_lower = cleaned_text.lower()
        result['has_abstract'] = 'abstract' in text_lower[:2000]
        result['has_introduction'] = 'introduction' in text_lower
        result['has_references'] = 'reference' in text_lower or 'bibliography' in text_lower
        
        # Detect formulas
        formulas = formula_processor.detect_formulas(cleaned_text)
        result['formulas_detected'] = len(formulas)
        
        # Count Greek letters
        greek_chars = 'αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'
        result['greek_letters'] = sum(1 for c in cleaned_text if c in greek_chars)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test extraction on batch of papers")
    parser.add_argument('--limit', type=int, default=200, help='Number of papers to test')
    parser.add_argument('--start-from', type=int, default=0, help='Start from this index')
    parser.add_argument('--output', type=str, default='test_extraction_results.json', help='Output file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Batch Extraction Test")
    print("=" * 80)
    print(f"Testing {args.limit} papers")
    print()
    
    # Load paper IDs
    paper_ids_file = Path("paper_ids.txt")
    if not paper_ids_file.exists():
        print("Error: paper_ids.txt not found")
        return
    
    with open(paper_ids_file, 'r') as f:
        all_paper_ids = [line.strip() for line in f if line.strip()]
    
    # Select sample
    paper_ids = all_paper_ids[args.start_from:args.start_from + args.limit]
    
    if not paper_ids:
        print("No papers to test")
        return
    
    # Initialize processors
    extractor = PDFExtractor(enable_ocr=False)
    processor = TextProcessor(
        remove_headers_footers=True,
        normalize_whitespace=True,
        improve_formulas=True
    )
    formula_processor = FormulaProcessor()
    
    pdf_dir = Path("pdfs")
    metadata_dir = Path("pdfs")
    
    # Process papers
    results = []
    stats = {
        'total': len(paper_ids),
        'successful': 0,
        'failed': 0,
        'extraction_methods': defaultdict(int),
        'extraction_times': [],
        'text_lengths': [],
        'formulas_total': 0,
        'greek_letters_total': 0,
        'structure_stats': {
            'has_abstract': 0,
            'has_introduction': 0,
            'has_references': 0
        }
    }
    
    start_time = time.time()
    
    for i, paper_id in enumerate(paper_ids, 1):
        if i % 10 == 0:
            print(f"[{i}/{len(paper_ids)}] Processing...", end='\r', flush=True)
        
        result = test_paper(paper_id, pdf_dir, metadata_dir, extractor, processor, formula_processor)
        results.append(result)
        
        if result['success']:
            stats['successful'] += 1
            stats['extraction_methods'][result['extraction_method']] += 1
            stats['extraction_times'].append(result['extraction_time'])
            stats['text_lengths'].append(result['text_length'])
            stats['formulas_total'] += result['formulas_detected']
            stats['greek_letters_total'] += result['greek_letters']
            
            if result['has_abstract']:
                stats['structure_stats']['has_abstract'] += 1
            if result['has_introduction']:
                stats['structure_stats']['has_introduction'] += 1
            if result['has_references']:
                stats['structure_stats']['has_references'] += 1
        else:
            stats['failed'] += 1
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if stats['extraction_times']:
        avg_time = sum(stats['extraction_times']) / len(stats['extraction_times'])
        total_extraction_time = sum(stats['extraction_times'])
    else:
        avg_time = total_extraction_time = 0
    
    if stats['text_lengths']:
        avg_length = sum(stats['text_lengths']) / len(stats['text_lengths'])
        min_length = min(stats['text_lengths'])
        max_length = max(stats['text_lengths'])
    else:
        avg_length = min_length = max_length = 0
    
    # Print report
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total Papers: {stats['total']}")
    print(f"Successful: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print()
    print("Extraction Methods:")
    for method, count in stats['extraction_methods'].items():
        print(f"  {method}: {count} ({count/stats['successful']*100:.1f}%)")
    print()
    print(f"Text Length:")
    print(f"  Average: {avg_length:,.0f} characters")
    print(f"  Range: {min_length:,} - {max_length:,} characters")
    print()
    print(f"Performance:")
    print(f"  Average Time: {avg_time:.2f} seconds per paper")
    print(f"  Total Time: {total_time:.1f} seconds")
    print(f"  Throughput: {stats['successful']/total_extraction_time*60:.1f} papers/minute")
    print()
    print(f"Formulas:")
    print(f"  Total Detected: {stats['formulas_total']}")
    print(f"  Average per Paper: {stats['formulas_total']/stats['successful']:.1f}")
    print(f"  Greek Letters: {stats['greek_letters_total']} ({stats['greek_letters_total']/stats['successful']:.1f} per paper)")
    print()
    print("Document Structure:")
    print(f"  Has Abstract: {stats['structure_stats']['has_abstract']} ({stats['structure_stats']['has_abstract']/stats['successful']*100:.1f}%)")
    print(f"  Has Introduction: {stats['structure_stats']['has_introduction']} ({stats['structure_stats']['has_introduction']/stats['successful']*100:.1f}%)")
    print(f"  Has References: {stats['structure_stats']['has_references']} ({stats['structure_stats']['has_references']/stats['successful']*100:.1f}%)")
    print("=" * 80)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_tested': len(paper_ids),
                'successful': stats['successful'],
                'failed': stats['failed'],
                'avg_extraction_time': avg_time,
                'avg_text_length': avg_length,
                'total_formulas': stats['formulas_total'],
                'total_greek_letters': stats['greek_letters_total']
            },
            'statistics': stats,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

