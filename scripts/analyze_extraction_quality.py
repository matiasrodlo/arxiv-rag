#!/usr/bin/env python3
"""
Analyze extraction results to identify weaknesses and plan improvements.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def analyze_results(results_file: str):
    """Analyze extraction results and identify weaknesses."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    stats = data['statistics']
    
    print("=" * 80)
    print("EXTRACTION QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Total Papers Analyzed: {len(results)}\n")
    
    # Categorize papers by quality
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)\n")
    
    if not successful:
        print("No successful extractions to analyze!")
        return
    
    # Analyze text length distribution
    text_lengths = [r.get('text_length', 0) for r in successful]
    avg_length = statistics.mean(text_lengths)
    median_length = statistics.median(text_lengths)
    min_length = min(text_lengths)
    max_length = max(text_lengths)
    
    print("=" * 80)
    print("TEXT LENGTH ANALYSIS")
    print("=" * 80)
    print(f"Average: {avg_length:,.0f} characters")
    print(f"Median: {median_length:,.0f} characters")
    print(f"Range: {min_length:,} - {max_length:,} characters")
    
    # Identify short texts (potential issues)
    short_texts = [r for r in successful if r.get('text_length', 0) < 10000]
    very_short = [r for r in successful if r.get('text_length', 0) < 5000]
    
    print(f"\nShort texts (<10K chars): {len(short_texts)} ({len(short_texts)/len(successful)*100:.1f}%)")
    print(f"Very short (<5K chars): {len(very_short)} ({len(very_short)/len(successful)*100:.1f}%)")
    
    if short_texts:
        print("\nShortest texts (potential extraction issues):")
        for r in sorted(short_texts, key=lambda x: x.get('text_length', 0))[:10]:
            print(f"  {r['paper_id']}: {r.get('text_length', 0):,} chars, {r.get('num_pages', 0)} pages")
    
    # Analyze page density
    print("\n" + "=" * 80)
    print("PAGE DENSITY ANALYSIS")
    print("=" * 80)
    
    densities = []
    low_density = []
    
    for r in successful:
        pages = r.get('num_pages', 1)
        length = r.get('text_length', 0)
        if pages > 0:
            density = length / pages
            densities.append(density)
            if density < 1000:
                low_density.append(r)
    
    if densities:
        avg_density = statistics.mean(densities)
        print(f"Average chars per page: {avg_density:,.0f}")
        print(f"Low density papers (<1K chars/page): {len(low_density)} ({len(low_density)/len(successful)*100:.1f}%)")
        
        if low_density:
            print("\nLow density papers (potential extraction issues):")
            for r in sorted(low_density, key=lambda x: x.get('text_length', 0) / max(x.get('num_pages', 1), 1))[:10]:
                density = r.get('text_length', 0) / max(r.get('num_pages', 1), 1)
                print(f"  {r['paper_id']}: {density:.0f} chars/page ({r.get('text_length', 0):,} chars, {r.get('num_pages', 0)} pages)")
    
    # Analyze document structure
    print("\n" + "=" * 80)
    print("DOCUMENT STRUCTURE ANALYSIS")
    print("=" * 80)
    
    has_abstract = sum(1 for r in successful if r.get('has_abstract', False))
    has_intro = sum(1 for r in successful if r.get('has_introduction', False))
    has_refs = sum(1 for r in successful if r.get('has_references', False))
    
    print(f"Has Abstract: {has_abstract} ({has_abstract/len(successful)*100:.1f}%)")
    print(f"Has Introduction: {has_intro} ({has_intro/len(successful)*100:.1f}%)")
    print(f"Has References: {has_refs} ({has_refs/len(successful)*100:.1f}%)")
    
    missing_structure = []
    for r in successful:
        missing = []
        if not r.get('has_abstract', False):
            missing.append('abstract')
        if not r.get('has_introduction', False):
            missing.append('introduction')
        if not r.get('has_references', False):
            missing.append('references')
        if missing:
            missing_structure.append((r, missing))
    
    if missing_structure:
        print(f"\nPapers missing structure: {len(missing_structure)} ({len(missing_structure)/len(successful)*100:.1f}%)")
        print("\nTop missing structure patterns:")
        patterns = Counter([tuple(missing) for _, missing in missing_structure])
        for pattern, count in patterns.most_common(5):
            print(f"  Missing {', '.join(pattern)}: {count} papers")
    
    # Analyze formula detection
    print("\n" + "=" * 80)
    print("FORMULA DETECTION ANALYSIS")
    print("=" * 80)
    
    formulas = [r.get('formulas_detected', 0) for r in successful]
    greek_letters = [r.get('greek_letters', 0) for r in successful]
    
    if formulas:
        avg_formulas = statistics.mean(formulas)
        print(f"Average formulas per paper: {avg_formulas:.1f}")
        print(f"Total formulas detected: {sum(formulas):,}")
    
    if greek_letters:
        avg_greek = statistics.mean(greek_letters)
        print(f"Average Greek letters per paper: {avg_greek:.1f}")
        print(f"Total Greek letters: {sum(greek_letters):,}")
    
    # Papers with Greek letters but no formulas (potential detection issue)
    formula_issues = [r for r in successful if r.get('greek_letters', 0) > 10 and r.get('formulas_detected', 0) == 0]
    if formula_issues:
        print(f"\nPotential formula detection issues:")
        print(f"  Papers with Greek letters but no formulas: {len(formula_issues)}")
        for r in formula_issues[:5]:
            print(f"    {r['paper_id']}: {r.get('greek_letters', 0)} Greek letters, 0 formulas")
    
    # Analyze extraction methods
    print("\n" + "=" * 80)
    print("EXTRACTION METHOD ANALYSIS")
    print("=" * 80)
    
    methods = Counter([r.get('extraction_method', 'unknown') for r in successful])
    for method, count in methods.most_common():
        print(f"{method}: {count} ({count/len(successful)*100:.1f}%)")
    
    # Analyze extraction times
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    times = [r.get('extraction_time', 0) for r in successful if r.get('extraction_time', 0) > 0]
    if times:
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        max_time = max(times)
        print(f"Average extraction time: {avg_time:.2f} seconds")
        print(f"Median extraction time: {median_time:.2f} seconds")
        print(f"Max extraction time: {max_time:.2f} seconds")
        
        # Slow extractions
        slow = [r for r in successful if r.get('extraction_time', 0) > 5.0]
        if slow:
            print(f"\nSlow extractions (>5s): {len(slow)} ({len(slow)/len(successful)*100:.1f}%)")
            for r in sorted(slow, key=lambda x: x.get('extraction_time', 0), reverse=True)[:5]:
                print(f"  {r['paper_id']}: {r.get('extraction_time', 0):.2f}s, {r.get('text_length', 0):,} chars")
    
    # Analyze failures
    if failed:
        print("\n" + "=" * 80)
        print("FAILURE ANALYSIS")
        print("=" * 80)
        
        errors = Counter([r.get('error', 'Unknown') for r in failed])
        for error, count in errors.most_common():
            print(f"{error}: {count}")
    
    # Identify weaknesses
    print("\n" + "=" * 80)
    print("IDENTIFIED WEAKNESSES")
    print("=" * 80)
    
    weaknesses = []
    
    if len(short_texts) > len(successful) * 0.05:  # More than 5%
        weaknesses.append({
            'issue': 'Short text extraction',
            'count': len(short_texts),
            'percentage': len(short_texts)/len(successful)*100,
            'severity': 'High' if len(short_texts) > len(successful) * 0.1 else 'Medium'
        })
    
    if len(low_density) > len(successful) * 0.05:
        weaknesses.append({
            'issue': 'Low page density',
            'count': len(low_density),
            'percentage': len(low_density)/len(successful)*100,
            'severity': 'High' if len(low_density) > len(successful) * 0.1 else 'Medium'
        })
    
    if len(missing_structure) > len(successful) * 0.1:  # More than 10%
        weaknesses.append({
            'issue': 'Missing document structure',
            'count': len(missing_structure),
            'percentage': len(missing_structure)/len(successful)*100,
            'severity': 'High' if len(missing_structure) > len(successful) * 0.2 else 'Medium'
        })
    
    if len(formula_issues) > 0:
        weaknesses.append({
            'issue': 'Formula detection issues',
            'count': len(formula_issues),
            'percentage': len(formula_issues)/len(successful)*100,
            'severity': 'Medium'
        })
    
    if len(failed) > len(results) * 0.01:  # More than 1% failure
        weaknesses.append({
            'issue': 'Extraction failures',
            'count': len(failed),
            'percentage': len(failed)/len(results)*100,
            'severity': 'High' if len(failed) > len(results) * 0.05 else 'Medium'
        })
    
    for weakness in weaknesses:
        print(f"\n{weakness['severity']} - {weakness['issue']}:")
        print(f"  Affected: {weakness['count']} papers ({weakness['percentage']:.1f}%)")
    
    return weaknesses, {
        'short_texts': short_texts,
        'low_density': low_density,
        'missing_structure': missing_structure,
        'formula_issues': formula_issues,
        'failed': failed
    }


def generate_improvement_plan(weaknesses, issues):
    """Generate improvement plan based on identified weaknesses."""
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT PLAN")
    print("=" * 80)
    
    improvements = []
    
    # Short text extraction
    if issues['short_texts']:
        improvements.append({
            'priority': 'High',
            'issue': 'Short text extraction',
            'description': f"{len(issues['short_texts'])} papers have very short extracted text",
            'solutions': [
                'Install PyMuPDF for better extraction quality',
                'Implement OCR fallback for scanned/image-based PDFs',
                'Add multi-strategy extraction (try multiple methods per page)',
                'Improve layout detection for multi-column papers'
            ],
            'impact': 'High - Improves text completeness'
        })
    
    # Low density
    if issues['low_density']:
        improvements.append({
            'priority': 'High',
            'issue': 'Low page density',
            'description': f"{len(issues['low_density'])} papers have low characters per page",
            'solutions': [
                'Use PyMuPDF layout mode for better extraction',
                'Implement image-based extraction for scanned pages',
                'Add table and figure extraction',
                'Improve handling of complex layouts'
            ],
            'impact': 'High - Improves content extraction'
        })
    
    # Missing structure
    if issues['missing_structure']:
        improvements.append({
            'priority': 'Medium',
            'issue': 'Missing document structure',
            'description': f"{len(issues['missing_structure'])} papers missing key sections",
            'solutions': [
                'Improve section detection algorithms',
                'Use NLP-based section identification',
                'Better handling of non-standard paper formats',
                'Add section boundary detection'
            ],
            'impact': 'Medium - Improves document understanding'
        })
    
    # Formula detection
    if issues['formula_issues']:
        improvements.append({
            'priority': 'Medium',
            'issue': 'Formula detection issues',
            'description': f"{len(issues['formula_issues'])} papers have Greek letters but no detected formulas",
            'solutions': [
                'Improve formula pattern matching',
                'Add LaTeX command detection',
                'Better handling of inline formulas',
                'Use ML-based formula detection'
            ],
            'impact': 'Medium - Improves formula handling'
        })
    
    # Failures
    if issues['failed']:
        improvements.append({
            'priority': 'High',
            'issue': 'Extraction failures',
            'description': f"{len(issues['failed'])} papers failed to extract",
            'solutions': [
                'Add better error handling and retry logic',
                'Implement OCR for corrupted PDFs',
                'Add PDF repair/preprocessing',
                'Better fallback mechanisms'
            ],
            'impact': 'High - Reduces failure rate'
        })
    
    # General improvements
    improvements.append({
        'priority': 'Medium',
        'issue': 'Extraction method diversity',
        'description': 'Currently only using pypdf (fallback method)',
        'solutions': [
            'Install PyMuPDF (primary, best quality)',
            'Install pdfplumber (better for tables)',
            'Use method selection based on PDF characteristics',
            'Implement hybrid extraction (combine methods)'
        ],
        'impact': 'High - Significantly improves quality'
    })
    
    for i, improvement in enumerate(improvements, 1):
        print(f"\n{i}. [{improvement['priority']}] {improvement['issue']}")
        print(f"   Description: {improvement['description']}")
        print(f"   Impact: {improvement['impact']}")
        print(f"   Solutions:")
        for solution in improvement['solutions']:
            print(f"     - {solution}")
    
    return improvements


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze extraction quality")
    parser.add_argument('--results', type=str, default='test_2000_results.json', help='Results file')
    parser.add_argument('--output', type=str, default='quality_analysis.json', help='Output analysis file')
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        return
    
    weaknesses, issues = analyze_results(args.results)
    improvements = generate_improvement_plan(weaknesses, issues)
    
    # Save analysis
    analysis = {
        'weaknesses': weaknesses,
        'issues_summary': {
            'short_texts': len(issues['short_texts']),
            'low_density': len(issues['low_density']),
            'missing_structure': len(issues['missing_structure']),
            'formula_issues': len(issues['formula_issues']),
            'failed': len(issues['failed'])
        },
        'improvements': improvements
    }
    
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis saved to: {args.output}")


if __name__ == "__main__":
    main()

