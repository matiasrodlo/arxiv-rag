#!/usr/bin/env python3
"""
Final comprehensive analysis for embedding model optimization.
Checks for any remaining issues or optimizations needed.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import statistics

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

def analyze_embedding_optimization(output_dir: Path, sample_size: int = None) -> Dict:
    """Comprehensive analysis for embedding optimization."""
    
    all_files = list(output_dir.rglob('*.json'))
    all_files = [f for f in all_files if not f.name.startswith('._')]
    
    if sample_size:
        all_files = all_files[:sample_size]
    
    print(f"Analyzing {len(all_files):,} files for final embedding optimization check...")
    print()
    
    # Initialize tokenizer
    tokenizer = None
    if TIKTOKEN_AVAILABLE:
        try:
            tokenizer = tiktoken.get_encoding('cl100k_base')
        except:
            pass
    
    stats = {
        'total_files': 0,
        'total_chunks': 0,
        
        # Size optimization
        'chunks_optimal_size': 0,  # 200-1500 chars
        'chunks_too_small': 0,  # < 50
        'chunks_small': 0,  # 50-200
        'chunks_large': 0,  # 1500-2000
        'chunks_too_large': 0,  # > 2000
        'chunk_sizes': [],
        
        # Token optimization
        'chunks_optimal_tokens': 0,  # 50-512 tokens
        'chunks_token_counts': [],
        'chunks_without_tokens': 0,
        'chunks_over_token_limit': 0,
        
        # Text quality
        'chunks_empty': 0,
        'chunks_whitespace_only': 0,
        'chunks_html_tags': 0,
        'chunks_special_chars_only': 0,
        'chunks_unicode_issues': 0,
        'chunks_excessive_newlines': 0,
        'chunks_excessive_spaces': 0,
        
        # Context optimization
        'chunks_with_context': 0,
        'chunks_without_context': 0,
        'chunks_with_paper_title': 0,
        'chunks_with_section': 0,
        
        # Metadata optimization
        'chunks_with_quality': 0,
        'chunks_without_quality': 0,
        'chunks_with_entities': 0,
        'chunks_without_entities': 0,
        'chunks_with_keywords': 0,
        'chunks_without_keywords': 0,
        'chunks_with_citations': 0,
        
        # Embedding-specific issues
        'chunks_url_only': 0,
        'chunks_citation_only': 0,
        'chunks_table_only': 0,
        'chunks_figure_only': 0,
        'chunks_equation_only': 0,
        'chunks_code_only': 0,
        
        # Consistency
        'chunks_length_mismatch': 0,
        'chunks_metadata_incomplete': 0,
        'chunks_duplicate_text': 0,
        
        # Advanced optimizations
        'chunks_needs_normalization': 0,
        'chunks_needs_cleaning': 0,
        'chunks_poor_quality': 0,  # quality < 0.8
        'chunks_excellent_quality': 0,  # quality >= 0.95
        
        'quality_scores': [],
        'errors': []
    }
    
    seen_texts = set()
    
    for i, file_path in enumerate(all_files):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} files...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats['total_files'] += 1
            chunks = data.get('chunks', [])
            stats['total_chunks'] += len(chunks)
            
            for chunk in chunks:
                text = chunk.get('text', '')
                meta = chunk.get('metadata', {})
                
                # Size analysis
                text_len = len(text)
                stats['chunk_sizes'].append(text_len)
                
                if text_len < 50:
                    stats['chunks_too_small'] += 1
                elif 50 <= text_len < 200:
                    stats['chunks_small'] += 1
                elif 200 <= text_len <= 1500:
                    stats['chunks_optimal_size'] += 1
                elif 1500 < text_len <= 2000:
                    stats['chunks_large'] += 1
                else:
                    stats['chunks_too_large'] += 1
                
                # Token analysis
                token_count = meta.get('token_count')
                if tokenizer and token_count is None:
                    try:
                        tokens = tokenizer.encode(text)
                        token_count = len(tokens)
                    except:
                        token_count = None
                
                if token_count is None:
                    stats['chunks_without_tokens'] += 1
                else:
                    stats['chunks_token_counts'].append(token_count)
                    if 50 <= token_count <= 512:
                        stats['chunks_optimal_tokens'] += 1
                    if token_count > 8192:
                        stats['chunks_over_token_limit'] += 1
                
                # Text quality
                if not text or not text.strip():
                    stats['chunks_empty'] += 1
                elif text.strip() == '':
                    stats['chunks_whitespace_only'] += 1
                elif re.match(r'^[\s\W]+$', text):
                    stats['chunks_special_chars_only'] += 1
                
                # HTML tags
                if re.search(r'<[^>]+>', text):
                    stats['chunks_html_tags'] += 1
                
                # Excessive formatting
                if text.count('\n') > len(text) / 10:
                    stats['chunks_excessive_newlines'] += 1
                if '  ' * 10 in text:
                    stats['chunks_excessive_spaces'] += 1
                
                # Context
                if text.startswith('#') or '##' in text[:100]:
                    stats['chunks_with_context'] += 1
                    if text.startswith('#'):
                        stats['chunks_with_paper_title'] += 1
                    if '##' in text:
                        stats['chunks_with_section'] += 1
                else:
                    stats['chunks_without_context'] += 1
                
                # Metadata
                if meta.get('quality_score') is not None:
                    stats['chunks_with_quality'] += 1
                    q_score = meta['quality_score']
                    stats['quality_scores'].append(q_score)
                    if q_score < 0.8:
                        stats['chunks_poor_quality'] += 1
                    elif q_score >= 0.95:
                        stats['chunks_excellent_quality'] += 1
                else:
                    stats['chunks_without_quality'] += 1
                
                if meta.get('entities'):
                    stats['chunks_with_entities'] += 1
                else:
                    stats['chunks_without_entities'] += 1
                
                if meta.get('keywords'):
                    stats['chunks_with_keywords'] += 1
                else:
                    stats['chunks_without_keywords'] += 1
                
                if meta.get('citation_count', 0) > 0:
                    stats['chunks_with_citations'] += 1
                
                # Embedding-specific content checks
                text_stripped = text.strip()
                if re.match(r'^https?://', text_stripped):
                    stats['chunks_url_only'] += 1
                elif re.match(r'^\[.*?\]$', text_stripped) or text_stripped.count('[') / max(len(text_stripped), 1) > 0.3:
                    stats['chunks_citation_only'] += 1
                elif re.match(r'^Table\s+\d+', text_stripped, re.I):
                    stats['chunks_table_only'] += 1
                elif re.match(r'^Figure\s+\d+', text_stripped, re.I):
                    stats['chunks_figure_only'] += 1
                elif re.match(r'^\$.*?\$', text_stripped):
                    stats['chunks_equation_only'] += 1
                elif re.match(r'^```|^def |^class ', text_stripped):
                    stats['chunks_code_only'] += 1
                
                # Consistency
                reported_length = meta.get('chunk_length', 0)
                if abs(reported_length - text_len) > 10:
                    stats['chunks_length_mismatch'] += 1
                
                # Duplicate detection
                text_hash = hash(text.strip().lower())
                if text_hash in seen_texts:
                    stats['chunks_duplicate_text'] += 1
                else:
                    seen_texts.add(text_hash)
                
                # Needs optimization flags
                needs_opt = False
                if re.search(r'<[^>]+>', text):
                    needs_opt = True
                if text.count('\n') > len(text) / 10:
                    needs_opt = True
                if not (text.startswith('#') or '##' in text[:100]):
                    needs_opt = True
                if meta.get('quality_score', 1) < 0.8:
                    needs_opt = True
                
                if needs_opt:
                    stats['chunks_needs_cleaning'] += 1
                    
        except json.JSONDecodeError as e:
            stats['errors'].append(f"{file_path.name}: JSON error")
        except Exception as e:
            stats['errors'].append(f"{file_path.name}: {type(e).__name__}")
    
    return stats

def print_final_analysis(stats: Dict):
    """Print comprehensive final analysis."""
    
    print("=" * 80)
    print("FINAL EMBEDDING OPTIMIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    total = max(stats['total_chunks'], 1)
    
    # Overall status
    print("OVERALL STATUS:")
    print(f"  Files analyzed: {stats['total_files']:,}")
    print(f"  Total chunks: {stats['total_chunks']:,}")
    print()
    
    # Size distribution
    print("SIZE DISTRIBUTION:")
    if stats['chunk_sizes']:
        print(f"  Average: {statistics.mean(stats['chunk_sizes']):.0f} chars")
        print(f"  Median: {statistics.median(stats['chunk_sizes']):.0f} chars")
        print(f"  Min: {min(stats['chunk_sizes'])} chars")
        print(f"  Max: {max(stats['chunk_sizes'])} chars")
        print()
        print(f"  Too small (<50): {stats['chunks_too_small']:,} ({stats['chunks_too_small']/total*100:.2f}%)")
        print(f"  Small (50-200): {stats['chunks_small']:,} ({stats['chunks_small']/total*100:.2f}%)")
        print(f"  ✅ Optimal (200-1500): {stats['chunks_optimal_size']:,} ({stats['chunks_optimal_size']/total*100:.2f}%)")
        print(f"  Large (1500-2000): {stats['chunks_large']:,} ({stats['chunks_large']/total*100:.2f}%)")
        print(f"  Too large (>2000): {stats['chunks_too_large']:,} ({stats['chunks_too_large']/total*100:.2f}%)")
    print()
    
    # Token distribution
    print("TOKEN DISTRIBUTION:")
    if stats['chunks_token_counts']:
        print(f"  Average: {statistics.mean(stats['chunks_token_counts']):.0f} tokens")
        print(f"  Median: {statistics.median(stats['chunks_token_counts']):.0f} tokens")
        print(f"  Max: {max(stats['chunks_token_counts'])} tokens")
        print(f"  ✅ Optimal (50-512): {stats['chunks_optimal_tokens']:,} ({stats['chunks_optimal_tokens']/len(stats['chunks_token_counts'])*100:.2f}%)")
        print(f"  Over limit (>8192): {stats['chunks_over_token_limit']:,}")
    print(f"  Without token_count: {stats['chunks_without_tokens']:,} ({stats['chunks_without_tokens']/total*100:.2f}%)")
    print()
    
    # Quality analysis
    print("QUALITY ANALYSIS:")
    if stats['quality_scores']:
        print(f"  Average: {statistics.mean(stats['quality_scores']):.3f}")
        print(f"  Min: {min(stats['quality_scores']):.3f}")
        print(f"  Max: {max(stats['quality_scores']):.3f}")
        print(f"  ✅ Excellent (>=0.95): {stats['chunks_excellent_quality']:,} ({stats['chunks_excellent_quality']/total*100:.2f}%)")
        print(f"  ⚠️  Poor (<0.8): {stats['chunks_poor_quality']:,} ({stats['chunks_poor_quality']/total*100:.2f}%)")
    print(f"  With quality score: {stats['chunks_with_quality']:,} ({stats['chunks_with_quality']/total*100:.2f}%)")
    print()
    
    # Text quality issues
    print("TEXT QUALITY ISSUES:")
    issues = []
    if stats['chunks_empty'] > 0:
        issues.append(f"  ⚠️  Empty: {stats['chunks_empty']:,} ({stats['chunks_empty']/total*100:.2f}%)")
    if stats['chunks_whitespace_only'] > 0:
        issues.append(f"  ⚠️  Whitespace-only: {stats['chunks_whitespace_only']:,} ({stats['chunks_whitespace_only']/total*100:.2f}%)")
    if stats['chunks_html_tags'] > 0:
        issues.append(f"  ⚠️  HTML tags: {stats['chunks_html_tags']:,} ({stats['chunks_html_tags']/total*100:.2f}%)")
    if stats['chunks_excessive_newlines'] > 0:
        issues.append(f"  ⚠️  Excessive newlines: {stats['chunks_excessive_newlines']:,} ({stats['chunks_excessive_newlines']/total*100:.2f}%)")
    if stats['chunks_excessive_spaces'] > 0:
        issues.append(f"  ⚠️  Excessive spaces: {stats['chunks_excessive_spaces']:,} ({stats['chunks_excessive_spaces']/total*100:.2f}%)")
    if not issues:
        print("  ✅ No text quality issues found")
    else:
        for issue in issues:
            print(issue)
    print()
    
    # Context optimization
    print("CONTEXT OPTIMIZATION:")
    print(f"  ✅ With context headers: {stats['chunks_with_context']:,} ({stats['chunks_with_context']/total*100:.2f}%)")
    print(f"  ⚠️  Without context: {stats['chunks_without_context']:,} ({stats['chunks_without_context']/total*100:.2f}%)")
    print(f"  With paper title: {stats['chunks_with_paper_title']:,} ({stats['chunks_with_paper_title']/total*100:.2f}%)")
    print(f"  With section: {stats['chunks_with_section']:,} ({stats['chunks_with_section']/total*100:.2f}%)")
    print()
    
    # Metadata coverage
    print("METADATA COVERAGE:")
    print(f"  Quality scores: {stats['chunks_with_quality']:,} ({stats['chunks_with_quality']/total*100:.2f}%)")
    print(f"  Entities: {stats['chunks_with_entities']:,} ({stats['chunks_with_entities']/total*100:.2f}%)")
    print(f"  Keywords: {stats['chunks_with_keywords']:,} ({stats['chunks_with_keywords']/total*100:.2f}%)")
    print(f"  Citations: {stats['chunks_with_citations']:,} ({stats['chunks_with_citations']/total*100:.2f}%)")
    print()
    
    # Embedding-specific content
    print("EMBEDDING-SPECIFIC CONTENT:")
    emb_issues = []
    if stats['chunks_url_only'] > 0:
        emb_issues.append(f"  ⚠️  URL-only: {stats['chunks_url_only']:,}")
    if stats['chunks_citation_only'] > 0:
        emb_issues.append(f"  ⚠️  Citation-only: {stats['chunks_citation_only']:,}")
    if stats['chunks_table_only'] > 0:
        emb_issues.append(f"  ⚠️  Table-only: {stats['chunks_table_only']:,}")
    if stats['chunks_equation_only'] > 0:
        emb_issues.append(f"  ⚠️  Equation-only: {stats['chunks_equation_only']:,}")
    if stats['chunks_code_only'] > 0:
        emb_issues.append(f"  ⚠️  Code-only: {stats['chunks_code_only']:,}")
    if not emb_issues:
        print("  ✅ No problematic content types found")
    else:
        for issue in emb_issues:
            print(issue)
    print()
    
    # Consistency
    print("CONSISTENCY:")
    if stats['chunks_length_mismatch'] > 0:
        print(f"  ⚠️  Length mismatches: {stats['chunks_length_mismatch']:,} ({stats['chunks_length_mismatch']/total*100:.2f}%)")
    if stats['chunks_duplicate_text'] > 0:
        print(f"  ⚠️  Duplicate text: {stats['chunks_duplicate_text']:,} ({stats['chunks_duplicate_text']/total*100:.2f}%)")
    if stats['chunks_length_mismatch'] == 0 and stats['chunks_duplicate_text'] == 0:
        print("  ✅ No consistency issues")
    print()
    
    # Recommendations
    print("=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    recommendations = []
    
    if stats['chunks_too_small'] > 0:
        recommendations.append(f"⚠️  HIGH: Filter {stats['chunks_too_small']:,} chunks < 50 chars - produce poor embeddings")
    
    if stats['chunks_too_large'] > 0:
        recommendations.append(f"⚠️  HIGH: Split {stats['chunks_too_large']:,} chunks > 2000 chars - may exceed limits")
    
    if stats['chunks_html_tags'] > 0:
        recommendations.append(f"⚠️  MEDIUM: Clean HTML tags from {stats['chunks_html_tags']:,} chunks")
    
    if stats['chunks_without_context'] > 0:
        recommendations.append(f"⚠️  MEDIUM: Add context headers to {stats['chunks_without_context']:,} chunks")
    
    if stats['chunks_excessive_newlines'] > 0:
        recommendations.append(f"⚠️  LOW: Normalize newlines in {stats['chunks_excessive_newlines']:,} chunks")
    
    if stats['chunks_poor_quality'] > 0:
        recommendations.append(f"⚠️  MEDIUM: Review {stats['chunks_poor_quality']:,} chunks with quality < 0.8")
    
    if stats['chunks_without_tokens'] > 0:
        recommendations.append(f"⚠️  LOW: Calculate token_count for {stats['chunks_without_tokens']:,} chunks")
    
    if stats['chunks_duplicate_text'] > 0:
        recommendations.append(f"⚠️  LOW: Review {stats['chunks_duplicate_text']:,} duplicate chunks")
    
    # Positive findings
    if stats['chunks_optimal_size'] / total > 0.8:
        recommendations.append(f"✅ Excellent: {stats['chunks_optimal_size']/total*100:.1f}% chunks in optimal size range")
    
    if stats['chunks_with_quality'] / total > 0.99:
        recommendations.append("✅ Excellent: 99%+ chunks have quality scores")
    
    if stats['chunks_with_context'] / total > 0.99:
        recommendations.append("✅ Excellent: 99%+ chunks have context headers")
    
    if stats['chunks_excellent_quality'] / total > 0.9:
        recommendations.append(f"✅ Excellent: {stats['chunks_excellent_quality']/total*100:.1f}% chunks have excellent quality (>=0.95)")
    
    if not recommendations:
        recommendations.append("✅ Chunks are fully optimized for embedding models!")
    
    for rec in recommendations:
        print(rec)
    
    print()
    print("=" * 80)
    
    # Overall assessment
    print()
    print("OVERALL ASSESSMENT:")
    issues_count = (
        stats['chunks_too_small'] + stats['chunks_too_large'] +
        stats['chunks_html_tags'] + stats['chunks_without_context'] +
        stats['chunks_poor_quality']
    )
    
    if issues_count == 0:
        print("✅ EXCELLENT - No optimization needed!")
        print("   Chunks are fully optimized for embedding models.")
    elif issues_count < total * 0.01:  # Less than 1%
        print("✅ VERY GOOD - Minor optimizations available")
        print(f"   Only {issues_count:,} chunks ({issues_count/total*100:.2f}%) need attention.")
    elif issues_count < total * 0.05:  # Less than 5%
        print("✅ GOOD - Some optimizations recommended")
        print(f"   {issues_count:,} chunks ({issues_count/total*100:.2f}%) could be improved.")
    else:
        print("⚠️  NEEDS OPTIMIZATION")
        print(f"   {issues_count:,} chunks ({issues_count/total*100:.2f}%) need attention.")
    
    print()
    print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Final embedding optimization analysis')
    parser.add_argument('input_dir', type=str, help='Input directory (output_improved)')
    parser.add_argument('--sample', type=int, default=None, help='Sample size (for testing)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found")
        return
    
    stats = analyze_embedding_optimization(input_dir, sample_size=args.sample)
    print_final_analysis(stats)
    
    # Save report
    report_file = Path('final_embedding_analysis_report.txt')
    import sys
    original_stdout = sys.stdout
    with open(report_file, 'w') as f:
        sys.stdout = f
        print_final_analysis(stats)
    sys.stdout = original_stdout
    
    print(f"\n✅ Detailed report saved to: {report_file}")

if __name__ == '__main__':
    main()
