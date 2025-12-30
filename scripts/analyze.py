#!/usr/bin/env python3
"""
Analyze JSON files and rate them 1-10 for RAG suitability.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def analyze_json_for_rag(json_path: Path) -> Dict:
    """Analyze a single JSON file and rate it for RAG suitability."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {
            'paper_id': json_path.stem,
            'error': str(e),
            'score': 0,
            'issues': [f'Failed to load: {str(e)}']
        }
    
    issues = []
    warnings = []
    strengths = []
    score = 10.0  # Start with perfect score, deduct for issues
    
    paper_id = data.get('paper_id', json_path.stem)
    
    # 1. Check required structure (critical for RAG)
    required_fields = ['paper_id', 'metadata', 'text', 'chunks']
    for field in required_fields:
        if field not in data:
            issues.append(f"Missing critical field: {field}")
            score -= 2.0
    
    # 2. Evaluate text quality
    text_data = data.get('text', {})
    full_text = text_data.get('full', '')
    text_length = len(full_text)
    
    if text_length < 500:
        issues.append(f"Text too short: {text_length} chars")
        score -= 2.0
    elif text_length < 2000:
        warnings.append(f"Text quite short: {text_length} chars")
        score -= 0.5
    else:
        strengths.append(f"Good text length: {text_length:,} chars")
    
    # Check text quality indicators
    if not full_text.strip():
        issues.append("Empty text")
        score -= 3.0
    
    # 3. Evaluate chunks (critical for RAG)
    chunks = data.get('chunks', [])
    num_chunks = len(chunks)
    
    if num_chunks == 0:
        issues.append("No chunks - cannot build RAG")
        score -= 3.0
    elif num_chunks < 5:
        warnings.append(f"Very few chunks: {num_chunks}")
        score -= 1.0
    elif num_chunks > 1000:
        warnings.append(f"Very many chunks: {num_chunks} (may be inefficient)")
        score -= 0.5
    else:
        strengths.append(f"Good chunk count: {num_chunks}")
    
    # Analyze chunk quality
    chunk_issues = 0
    chunk_metadata_issues = 0
    avg_chunk_size = 0
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        chunk_meta = chunk.get('metadata', {})
        
        if not chunk_text.strip():
            chunk_issues += 1
        else:
            avg_chunk_size += len(chunk_text)
        
        # Check chunk metadata (important for RAG context)
        if 'chunk_index' not in chunk_meta:
            chunk_metadata_issues += 1
        if 'paper_id' not in chunk_meta:
            chunk_metadata_issues += 1
        if 'section' not in chunk_meta:
            chunk_metadata_issues += 1
        if 'page' not in chunk_meta:
            chunk_metadata_issues += 1
    
    if chunk_issues > 0:
        issues.append(f"{chunk_issues} empty chunks")
        score -= min(1.0, chunk_issues * 0.1)
    
    if chunk_metadata_issues > num_chunks * 0.1:  # More than 10% missing metadata
        warnings.append(f"Many chunks missing metadata: {chunk_metadata_issues}/{num_chunks}")
        score -= 0.5
    
    if num_chunks > 0:
        avg_chunk_size = avg_chunk_size / num_chunks
        if avg_chunk_size < 100:
            warnings.append(f"Chunks too small: avg {avg_chunk_size:.0f} chars")
            score -= 0.5
        elif avg_chunk_size > 2000:
            warnings.append(f"Chunks too large: avg {avg_chunk_size:.0f} chars")
            score -= 0.3
        else:
            strengths.append(f"Good chunk size: avg {avg_chunk_size:.0f} chars")
    
    # 4. Evaluate sections (helpful for RAG context)
    sections = text_data.get('sections', [])
    num_sections = len(sections)
    
    if num_sections == 0:
        warnings.append("No sections extracted")
        score -= 0.5
    elif num_sections < 3:
        warnings.append(f"Few sections: {num_sections}")
        score -= 0.3
    else:
        strengths.append(f"Good section structure: {num_sections} sections")
    
    # Check section quality
    section_issues = 0
    for section in sections:
        if not section.get('name', '').strip():
            section_issues += 1
        if not section.get('text', '').strip():
            section_issues += 1
    
    if section_issues > num_sections * 0.2:
        warnings.append(f"Many empty sections: {section_issues}/{num_sections}")
        score -= 0.3
    
    # 5. Evaluate page-level data (useful for RAG)
    pages = text_data.get('by_page', [])
    num_pages = len(pages)
    
    if num_pages == 0:
        warnings.append("No page-level data")
        score -= 0.3
    else:
        strengths.append(f"Page-level data available: {num_pages} pages")
    
    # 6. Evaluate metadata (important for RAG filtering)
    metadata = data.get('metadata', {})
    
    metadata_fields = ['paper_id', 'extraction_method', 'quality_score', 'pdf_type', 'num_pages']
    missing_metadata = [f for f in metadata_fields if f not in metadata]
    
    if missing_metadata:
        warnings.append(f"Missing metadata fields: {', '.join(missing_metadata)}")
        score -= len(missing_metadata) * 0.2
    
    quality_score = metadata.get('quality_score', 0.0)
    if quality_score < 0.7:
        warnings.append(f"Low extraction quality score: {quality_score:.2f}")
        score -= 0.5
    elif quality_score >= 0.9:
        strengths.append(f"High quality extraction: {quality_score:.2f}")
    
    # 7. Evaluate statistics (useful for monitoring)
    stats = data.get('statistics', {})
    if not stats:
        warnings.append("No statistics provided")
        score -= 0.2
    
    # 8. Check for duplicate/redundant content
    if num_chunks > 0:
        chunk_texts = [chunk.get('text', '').strip() for chunk in chunks]
        unique_chunks = len(set(chunk_texts))
        if unique_chunks < num_chunks * 0.9:
            warnings.append(f"Potential duplicate chunks: {num_chunks - unique_chunks} duplicates")
            score -= 0.3
    
    # Ensure score is between 0 and 10
    score = max(0.0, min(10.0, score))
    
    return {
        'paper_id': paper_id,
        'score': round(score, 1),
        'text_length': text_length,
        'num_chunks': num_chunks,
        'num_sections': num_sections,
        'num_pages': num_pages,
        'avg_chunk_size': round(avg_chunk_size, 0) if num_chunks > 0 else 0,
        'quality_score': quality_score,
        'issues': issues,
        'warnings': warnings,
        'strengths': strengths
    }

def main():
    json_dir = Path('output')
    
    if not json_dir.exists():
        print(f"Error: {json_dir} directory not found")
        sys.exit(1)
    
    json_files = [f for f in json_dir.glob('*.json') if not f.name.startswith('._')]
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        sys.exit(1)
    
    print(f"Analyzing {len(json_files)} JSON files for RAG suitability...\n")
    
    results = []
    score_distribution = defaultdict(int)
    
    for json_file in sorted(json_files):
        result = analyze_json_for_rag(json_file)
        results.append(result)
        score_bucket = int(result['score'])
        score_distribution[score_bucket] += 1
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print summary
    print("=" * 80)
    print("RAG SUITABILITY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal files analyzed: {len(results)}")
    print(f"Average score: {sum(r['score'] for r in results) / len(results):.1f}/10")
    print(f"Highest score: {max(r['score'] for r in results):.1f}/10")
    print(f"Lowest score: {min(r['score'] for r in results):.1f}/10")
    
    print("\nScore Distribution:")
    for score in sorted(score_distribution.keys(), reverse=True):
        count = score_distribution[score]
        bar = '█' * (count * 50 // len(results))
        print(f"  {score}/10: {count:3d} files {bar}")
    
    # Filter out error results for display
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    # Print top and bottom performers
    print("\n" + "=" * 80)
    print("TOP 10 FILES (Best for RAG)")
    print("=" * 80)
    for i, result in enumerate(valid_results[:10], 1):
        print(f"\n{i}. {result['paper_id']} - Score: {result['score']}/10")
        print(f"   Chunks: {result.get('num_chunks', 0)}, Sections: {result.get('num_sections', 0)}, "
              f"Text: {result.get('text_length', 0):,} chars")
        if result.get('strengths'):
            print(f"   Strengths: {', '.join(result['strengths'][:2])}")
        if result.get('warnings'):
            print(f"   Warnings: {', '.join(result['warnings'][:2])}")
    
    if valid_results:
        print("\n" + "=" * 80)
        print("BOTTOM 10 FILES (Need Improvement)")
        print("=" * 80)
        for i, result in enumerate(valid_results[-10:], 1):
            print(f"\n{i}. {result['paper_id']} - Score: {result['score']}/10")
            print(f"   Chunks: {result.get('num_chunks', 0)}, Sections: {result.get('num_sections', 0)}, "
                  f"Text: {result.get('text_length', 0):,} chars")
            if result.get('issues'):
                print(f"   Issues: {', '.join(result['issues'][:3])}")
            if result.get('warnings'):
                print(f"   Warnings: {', '.join(result['warnings'][:2])}")
    
    if error_results:
        print("\n" + "=" * 80)
        print(f"FILES WITH ERRORS ({len(error_results)} files)")
        print("=" * 80)
        for result in error_results[:10]:
            print(f"  {result['paper_id']}: {result.get('error', 'Unknown error')}")
    
    # Common issues analysis
    print("\n" + "=" * 80)
    print("COMMON ISSUES")
    print("=" * 80)
    all_issues = defaultdict(int)
    all_warnings = defaultdict(int)
    
    for result in results:
        if 'error' not in result:
            for issue in result.get('issues', []):
                all_issues[issue] += 1
            for warning in result.get('warnings', []):
                all_warnings[warning] += 1
    
    if all_issues:
        print("\nCritical Issues:")
        for issue, count in sorted(all_issues.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {issue}: {count} files")
    
    if all_warnings:
        print("\nCommon Warnings:")
        for warning, count in sorted(all_warnings.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {warning}: {count} files")
    
    # Save detailed results
    output_file = Path('json_rag_analysis_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_files': len(results),
                'average_score': round(sum(r['score'] for r in results) / len(results), 1),
                'max_score': max(r['score'] for r in results),
                'min_score': min(r['score'] for r in results),
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Recommendations
    valid_results = [r for r in results if 'error' not in r]
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR RAG")
    print("=" * 80)
    
    if valid_results:
        low_score_count = sum(1 for r in valid_results if r.get('score', 0) < 7.0)
        if low_score_count > 0:
            print(f"\n⚠ {low_score_count} files scored below 7.0 - consider reprocessing")
        
        no_chunks = sum(1 for r in valid_results if r.get('num_chunks', 0) == 0)
        if no_chunks > 0:
            print(f"⚠ {no_chunks} files have no chunks - cannot use for RAG")
        
        short_text = sum(1 for r in valid_results if r.get('text_length', 0) < 2000)
        if short_text > 0:
            print(f"⚠ {short_text} files have very short text - may have extraction issues")
        
        high_quality = sum(1 for r in valid_results if r.get('score', 0) >= 8.0)
        print(f"\n✓ {high_quality} files scored 8.0+ - excellent for RAG")
        print(f"✓ {len(valid_results) - low_score_count} files scored 7.0+ - good for RAG")
    
    if error_results:
        print(f"\n⚠ {len(error_results)} files had errors and could not be analyzed")

if __name__ == '__main__':
    main()

