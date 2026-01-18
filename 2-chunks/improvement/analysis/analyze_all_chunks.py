#!/usr/bin/env python3
"""
Comprehensive analysis of all optimized chunks.
Identifies potential improvements and issues.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import statistics

def analyze_all_files(output_dir: Path) -> Dict:
    """Analyze all JSON files in output_improved."""
    
    all_files = list(output_dir.rglob('*.json'))
    all_files = [f for f in all_files if not f.name.startswith('._')]
    
    print(f"Analyzing {len(all_files):,} files...")
    print()
    
    stats = {
        'total_files': 0,
        'files_with_chunks': 0,
        'files_without_chunks': 0,
        'total_chunks': 0,
        'chunks_with_metadata': 0,
        'chunks_without_metadata': 0,
        'chunks_with_quality': 0,
        'chunks_without_quality': 0,
        'chunks_with_entities': 0,
        'chunks_with_keywords': 0,
        'chunks_with_citations': 0,
        'chunks_with_section': 0,
        'chunks_with_context_headers': 0,
        'chunks_empty_text': 0,
        'chunks_very_short': 0,  # < 50 chars
        'chunks_very_long': 0,   # > 2000 chars
        'quality_scores': [],
        'chunk_sizes': [],
        'metadata_keys_frequency': Counter(),
        'missing_metadata_keys': defaultdict(int),
        'section_names': Counter(),
        'errors': [],
        'files_with_errors': 0,
        'inconsistent_structures': 0,
    }
    
    required_metadata_keys = {
        'paper_id', 'chunk_index', 'quality_score', 'section',
        'char_start', 'char_end', 'chunk_length'
    }
    
    for i, file_path in enumerate(all_files):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} files...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats['total_files'] += 1
            
            chunks = data.get('chunks', [])
            if not chunks:
                stats['files_without_chunks'] += 1
                continue
            
            stats['files_with_chunks'] += 1
            stats['total_chunks'] += len(chunks)
            
            # Analyze each chunk
            for chunk in chunks:
                text = chunk.get('text', '')
                meta = chunk.get('metadata', {})
                
                # Text analysis
                if not text or not text.strip():
                    stats['chunks_empty_text'] += 1
                elif len(text) < 50:
                    stats['chunks_very_short'] += 1
                elif len(text) > 2000:
                    stats['chunks_very_long'] += 1
                
                stats['chunk_sizes'].append(len(text))
                
                # Metadata analysis
                if meta:
                    stats['chunks_with_metadata'] += 1
                    
                    # Check for required keys
                    for key in required_metadata_keys:
                        if key not in meta:
                            stats['missing_metadata_keys'][key] += 1
                    
                    # Track metadata key frequency
                    for key in meta.keys():
                        stats['metadata_keys_frequency'][key] += 1
                    
                    # Quality score
                    if 'quality_score' in meta and meta['quality_score'] is not None:
                        stats['chunks_with_quality'] += 1
                        stats['quality_scores'].append(meta['quality_score'])
                    else:
                        stats['chunks_without_quality'] += 1
                    
                    # Features
                    if meta.get('entities'):
                        stats['chunks_with_entities'] += 1
                    if meta.get('keywords'):
                        stats['chunks_with_keywords'] += 1
                    if meta.get('citation_count', 0) > 0:
                        stats['chunks_with_citations'] += 1
                    if meta.get('section'):
                        stats['chunks_with_section'] += 1
                        stats['section_names'][meta['section']] += 1
                    
                    # Context headers
                    if text.startswith('#') or '##' in text:
                        stats['chunks_with_context_headers'] += 1
                else:
                    stats['chunks_without_metadata'] += 1
                    
        except json.JSONDecodeError as e:
            stats['errors'].append(f"{file_path.name}: JSON decode error - {str(e)[:50]}")
            stats['files_with_errors'] += 1
        except Exception as e:
            stats['errors'].append(f"{file_path.name}: {type(e).__name__} - {str(e)[:50]}")
            stats['files_with_errors'] += 1
    
    return stats

def print_analysis(stats: Dict):
    """Print comprehensive analysis results."""
    
    print("=" * 80)
    print("COMPREHENSIVE CHUNK ANALYSIS")
    print("=" * 80)
    print()
    
    # File statistics
    print("FILE STATISTICS:")
    print(f"  Total files: {stats['total_files']:,}")
    print(f"  Files with chunks: {stats['files_with_chunks']:,}")
    print(f"  Files without chunks: {stats['files_without_chunks']:,}")
    if stats['files_with_errors'] > 0:
        print(f"  ⚠️  Files with errors: {stats['files_with_errors']:,}")
    print()
    
    # Chunk statistics
    print("CHUNK STATISTICS:")
    print(f"  Total chunks: {stats['total_chunks']:,}")
    print(f"  Chunks with metadata: {stats['chunks_with_metadata']:,} ({stats['chunks_with_metadata']/max(stats['total_chunks'],1)*100:.1f}%)")
    print(f"  Chunks without metadata: {stats['chunks_without_metadata']:,}")
    print()
    
    # Quality analysis
    print("QUALITY ANALYSIS:")
    if stats['quality_scores']:
        print(f"  Chunks with quality scores: {stats['chunks_with_quality']:,} ({stats['chunks_with_quality']/max(stats['total_chunks'],1)*100:.1f}%)")
        print(f"  Chunks without quality scores: {stats['chunks_without_quality']:,}")
        print(f"  Average quality: {statistics.mean(stats['quality_scores']):.3f}")
        print(f"  Min quality: {min(stats['quality_scores']):.3f}")
        print(f"  Max quality: {max(stats['quality_scores']):.3f}")
        low_quality = sum(1 for q in stats['quality_scores'] if q < 0.7)
        if low_quality > 0:
            print(f"  ⚠️  Low quality chunks (<0.7): {low_quality:,}")
    else:
        print("  ⚠️  No quality scores found!")
    print()
    
    # Feature coverage
    print("FEATURE COVERAGE:")
    total = max(stats['total_chunks'], 1)
    print(f"  Entities: {stats['chunks_with_entities']:,} ({stats['chunks_with_entities']/total*100:.1f}%)")
    print(f"  Keywords: {stats['chunks_with_keywords']:,} ({stats['chunks_with_keywords']/total*100:.1f}%)")
    print(f"  Citations: {stats['chunks_with_citations']:,} ({stats['chunks_with_citations']/total*100:.1f}%)")
    print(f"  Section info: {stats['chunks_with_section']:,} ({stats['chunks_with_section']/total*100:.1f}%)")
    print(f"  Context headers: {stats['chunks_with_context_headers']:,} ({stats['chunks_with_context_headers']/total*100:.1f}%)")
    print()
    
    # Text quality issues
    print("TEXT QUALITY ISSUES:")
    if stats['chunks_empty_text'] > 0:
        print(f"  ⚠️  Empty text chunks: {stats['chunks_empty_text']:,}")
    if stats['chunks_very_short'] > 0:
        print(f"  ⚠️  Very short chunks (<50 chars): {stats['chunks_very_short']:,}")
    if stats['chunks_very_long'] > 0:
        print(f"  ⚠️  Very long chunks (>2000 chars): {stats['chunks_very_long']:,}")
    if stats['chunks_empty_text'] == 0 and stats['chunks_very_short'] == 0 and stats['chunks_very_long'] == 0:
        print("  ✅ No text quality issues found")
    print()
    
    # Chunk size statistics
    if stats['chunk_sizes']:
        print("CHUNK SIZE STATISTICS:")
        print(f"  Average: {statistics.mean(stats['chunk_sizes']):.0f} chars")
        print(f"  Median: {statistics.median(stats['chunk_sizes']):.0f} chars")
        print(f"  Min: {min(stats['chunk_sizes'])} chars")
        print(f"  Max: {max(stats['chunk_sizes'])} chars")
        print()
    
    # Missing metadata keys
    if stats['missing_metadata_keys']:
        print("⚠️  MISSING METADATA KEYS:")
        for key, count in sorted(stats['missing_metadata_keys'].items(), key=lambda x: x[1], reverse=True):
            pct = count / max(stats['total_chunks'], 1) * 100
            print(f"  {key}: {count:,} chunks ({pct:.1f}%)")
        print()
    else:
        print("✅ All required metadata keys present")
        print()
    
    # Most common metadata keys
    print("TOP 20 METADATA KEYS (by frequency):")
    for key, count in stats['metadata_keys_frequency'].most_common(20):
        pct = count / max(stats['total_chunks'], 1) * 100
        print(f"  {key}: {count:,} ({pct:.1f}%)")
    print()
    
    # Section distribution
    if stats['section_names']:
        print("TOP 10 SECTIONS:")
        for section, count in stats['section_names'].most_common(10):
            print(f"  {section}: {count:,} chunks")
        print()
    
    # Errors
    if stats['errors']:
        print(f"⚠️  ERRORS FOUND ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
        print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    recommendations = []
    
    if stats['chunks_without_quality'] > 0:
        recommendations.append(f"⚠️  {stats['chunks_without_quality']:,} chunks missing quality scores - consider adding default scores")
    
    if stats['chunks_empty_text'] > 0:
        recommendations.append(f"⚠️  {stats['chunks_empty_text']:,} chunks with empty text - should be filtered out")
    
    if stats['chunks_very_short'] > 0:
        recommendations.append(f"⚠️  {stats['chunks_very_short']:,} very short chunks (<50 chars) - may need filtering")
    
    if stats['chunks_very_long'] > 0:
        recommendations.append(f"⚠️  {stats['chunks_very_long']:,} very long chunks (>2000 chars) - may need splitting")
    
    if stats['missing_metadata_keys']:
        recommendations.append(f"⚠️  Some chunks missing required metadata keys - check data consistency")
    
    if stats['files_with_errors'] > 0:
        recommendations.append(f"⚠️  {stats['files_with_errors']} files with errors - need investigation")
    
    if stats['chunks_without_metadata'] > 0:
        recommendations.append(f"⚠️  {stats['chunks_without_metadata']:,} chunks without metadata - should have metadata")
    
    # Positive findings
    if stats['chunks_with_quality'] / max(stats['total_chunks'], 1) > 0.95:
        recommendations.append(f"✅ Excellent quality score coverage ({stats['chunks_with_quality']/max(stats['total_chunks'],1)*100:.1f}%)")
    
    if stats['chunks_with_entities'] / max(stats['total_chunks'], 1) > 0.9:
        recommendations.append(f"✅ Excellent entity extraction coverage ({stats['chunks_with_entities']/max(stats['total_chunks'],1)*100:.1f}%)")
    
    if stats['chunks_with_keywords'] / max(stats['total_chunks'], 1) > 0.9:
        recommendations.append(f"✅ Excellent keyword extraction coverage ({stats['chunks_with_keywords']/max(stats['total_chunks'],1)*100:.1f}%)")
    
    if not recommendations:
        recommendations.append("✅ No major issues found - chunks are well-optimized!")
    
    for rec in recommendations:
        print(rec)
    
    print()
    print("=" * 80)

def main():
    output_dir = Path('output_improved')
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} not found")
        return
    
    stats = analyze_all_files(output_dir)
    print_analysis(stats)
    
    # Save detailed report
    report_file = Path('chunk_analysis_report.txt')
    import sys
    original_stdout = sys.stdout
    with open(report_file, 'w') as f:
        sys.stdout = f
        print_analysis(stats)
    sys.stdout = original_stdout
    
    print(f"\n✅ Detailed report saved to: {report_file}")

if __name__ == '__main__':
    main()
