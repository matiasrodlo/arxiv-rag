#!/usr/bin/env python3
"""
Comprehensive analysis of chunks for embedding model optimization.
Checks for issues that could affect embedding quality.
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
    print("Warning: tiktoken not available. Token analysis will be limited.")

def analyze_for_embeddings(output_dir: Path, sample_size: int = None) -> Dict:
    """Analyze all chunks for embedding model optimization."""
    
    all_files = list(output_dir.rglob('*.json'))
    all_files = [f for f in all_files if not f.name.startswith('._')]
    
    if sample_size:
        all_files = all_files[:sample_size]
    
    print(f"Analyzing {len(all_files):,} files for embedding optimization...")
    print()
    
    # Initialize tokenizer if available
    tokenizer = None
    if TIKTOKEN_AVAILABLE:
        try:
            tokenizer = tiktoken.get_encoding('cl100k_base')  # OpenAI's tokenizer
        except:
            pass
    
    stats = {
        'total_files': 0,
        'total_chunks': 0,
        'chunks_analyzed': 0,
        
        # Size issues
        'chunks_too_small': 0,  # < 50 chars
        'chunks_too_large': 0,  # > 2000 chars
        'chunks_optimal_size': 0,  # 200-1500 chars
        'chunk_sizes': [],
        
        # Token issues
        'chunks_without_tokens': 0,
        'chunks_token_count_none': 0,
        'token_counts': [],
        'chunks_over_token_limit': 0,  # > 8192 tokens (common limit)
        
        # Text quality issues
        'chunks_empty': 0,
        'chunks_whitespace_only': 0,
        'chunks_special_chars_only': 0,
        'chunks_encoding_issues': 0,
        'chunks_unicode_issues': 0,
        
        # Context issues
        'chunks_without_context_headers': 0,
        'chunks_without_section': 0,
        'chunks_without_paper_id': 0,
        
        # Metadata issues
        'chunks_without_quality': 0,
        'chunks_without_entities': 0,
        'chunks_without_keywords': 0,
        
        # Embedding-specific issues
        'chunks_with_html_tags': 0,
        'chunks_with_latex_only': 0,
        'chunks_with_excessive_newlines': 0,
        'chunks_with_excessive_spaces': 0,
        'chunks_with_urls_only': 0,
        'chunks_with_citations_only': 0,
        
        # Consistency issues
        'chunks_length_mismatch': 0,
        'chunks_metadata_incomplete': 0,
        
        'errors': []
    }
    
    # Optimal ranges for different embedding models
    optimal_ranges = {
        'openai': (50, 8192),  # tokens
        'cohere': (50, 2048),  # tokens
        'sentence_transformers': (50, 512),  # tokens
        'universal': (200, 1500),  # chars
    }
    
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
                stats['chunks_analyzed'] += 1
                text = chunk.get('text', '')
                meta = chunk.get('metadata', {})
                
                # Size analysis
                text_len = len(text)
                stats['chunk_sizes'].append(text_len)
                
                if text_len < 50:
                    stats['chunks_too_small'] += 1
                elif text_len > 2000:
                    stats['chunks_too_large'] += 1
                elif 200 <= text_len <= 1500:
                    stats['chunks_optimal_size'] += 1
                
                # Token analysis
                token_count = meta.get('token_count')
                if tokenizer and token_count is None:
                    # Calculate tokens
                    try:
                        tokens = tokenizer.encode(text)
                        token_count = len(tokens)
                    except:
                        token_count = None
                
                if token_count is None:
                    stats['chunks_token_count_none'] += 1
                else:
                    stats['token_counts'].append(token_count)
                    if token_count > 8192:  # Common embedding model limit
                        stats['chunks_over_token_limit'] += 1
                
                # Text quality checks
                if not text or not text.strip():
                    stats['chunks_empty'] += 1
                elif text.strip() == '':
                    stats['chunks_whitespace_only'] += 1
                elif re.match(r'^[\s\W]+$', text):  # Only special chars/whitespace
                    stats['chunks_special_chars_only'] += 1
                
                # Encoding issues
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    stats['chunks_encoding_issues'] += 1
                
                # Check for problematic patterns
                if re.search(r'<[^>]+>', text):  # HTML tags
                    stats['chunks_with_html_tags'] += 1
                
                if re.match(r'^[\$\\]+', text):  # LaTeX only
                    stats['chunks_with_latex_only'] += 1
                
                if text.count('\n') > text_len / 10:  # Excessive newlines
                    stats['chunks_with_excessive_newlines'] += 1
                
                if '  ' * 10 in text:  # Excessive spaces
                    stats['chunks_with_excessive_spaces'] += 1
                
                if re.match(r'^https?://', text.strip()):  # URL only
                    stats['chunks_with_urls_only'] += 1
                
                # Citation-only check (very high citation ratio)
                citations = meta.get('citations', [])
                if citations and len(''.join(str(c) for c in citations)) / max(text_len, 1) > 0.5:
                    stats['chunks_with_citations_only'] += 1
                
                # Context checks
                if not (text.startswith('#') or '##' in text):
                    stats['chunks_without_context_headers'] += 1
                
                if not meta.get('section'):
                    stats['chunks_without_section'] += 1
                
                if not meta.get('paper_id'):
                    stats['chunks_without_paper_id'] += 1
                
                # Metadata checks
                if meta.get('quality_score') is None:
                    stats['chunks_without_quality'] += 1
                
                if not meta.get('entities'):
                    stats['chunks_without_entities'] += 1
                
                if not meta.get('keywords'):
                    stats['chunks_without_keywords'] += 1
                
                # Consistency checks
                reported_length = meta.get('chunk_length', 0)
                if abs(reported_length - text_len) > 10:
                    stats['chunks_length_mismatch'] += 1
                
                # Required metadata for embeddings
                required_keys = ['paper_id', 'chunk_index', 'quality_score', 'section']
                if not all(k in meta for k in required_keys):
                    stats['chunks_metadata_incomplete'] += 1
                    
        except json.JSONDecodeError as e:
            stats['errors'].append(f"{file_path.name}: JSON error")
        except Exception as e:
            stats['errors'].append(f"{file_path.name}: {type(e).__name__}")
    
    return stats

def print_embedding_analysis(stats: Dict):
    """Print comprehensive embedding optimization analysis."""
    
    print("=" * 80)
    print("EMBEDDING MODEL OPTIMIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    total = max(stats['chunks_analyzed'], 1)
    
    # File statistics
    print("FILE STATISTICS:")
    print(f"  Files analyzed: {stats['total_files']:,}")
    print(f"  Total chunks: {stats['total_chunks']:,}")
    print(f"  Chunks analyzed: {stats['chunks_analyzed']:,}")
    print()
    
    # Size analysis
    print("CHUNK SIZE ANALYSIS:")
    if stats['chunk_sizes']:
        print(f"  Average size: {statistics.mean(stats['chunk_sizes']):.0f} chars")
        print(f"  Median size: {statistics.median(stats['chunk_sizes']):.0f} chars")
        print(f"  Min size: {min(stats['chunk_sizes'])} chars")
        print(f"  Max size: {max(stats['chunk_sizes'])} chars")
        print()
        print(f"  Too small (<50 chars): {stats['chunks_too_small']:,} ({stats['chunks_too_small']/total*100:.2f}%)")
        print(f"  Too large (>2000 chars): {stats['chunks_too_large']:,} ({stats['chunks_too_large']/total*100:.2f}%)")
        print(f"  Optimal (200-1500 chars): {stats['chunks_optimal_size']:,} ({stats['chunks_optimal_size']/total*100:.2f}%)")
    print()
    
    # Token analysis
    print("TOKEN ANALYSIS:")
    if stats['token_counts']:
        print(f"  Average tokens: {statistics.mean(stats['token_counts']):.0f}")
        print(f"  Median tokens: {statistics.median(stats['token_counts']):.0f}")
        print(f"  Max tokens: {max(stats['token_counts'])}")
        print(f"  Chunks over 8192 tokens: {stats['chunks_over_token_limit']:,} ({stats['chunks_over_token_limit']/len(stats['token_counts'])*100:.2f}%)")
    else:
        print("  ⚠️  No token counts available")
    print(f"  Chunks without token_count: {stats['chunks_token_count_none']:,} ({stats['chunks_token_count_none']/total*100:.2f}%)")
    print()
    
    # Text quality issues
    print("TEXT QUALITY ISSUES:")
    issues_found = False
    if stats['chunks_empty'] > 0:
        print(f"  ⚠️  Empty chunks: {stats['chunks_empty']:,} ({stats['chunks_empty']/total*100:.2f}%)")
        issues_found = True
    if stats['chunks_whitespace_only'] > 0:
        print(f"  ⚠️  Whitespace-only: {stats['chunks_whitespace_only']:,} ({stats['chunks_whitespace_only']/total*100:.2f}%)")
        issues_found = True
    if stats['chunks_special_chars_only'] > 0:
        print(f"  ⚠️  Special chars only: {stats['chunks_special_chars_only']:,} ({stats['chunks_special_chars_only']/total*100:.2f}%)")
        issues_found = True
    if stats['chunks_encoding_issues'] > 0:
        print(f"  ⚠️  Encoding issues: {stats['chunks_encoding_issues']:,} ({stats['chunks_encoding_issues']/total*100:.2f}%)")
        issues_found = True
    if not issues_found:
        print("  ✅ No major text quality issues")
    print()
    
    # Embedding-specific issues
    print("EMBEDDING-SPECIFIC ISSUES:")
    emb_issues = False
    if stats['chunks_with_html_tags'] > 0:
        print(f"  ⚠️  HTML tags present: {stats['chunks_with_html_tags']:,} ({stats['chunks_with_html_tags']/total*100:.2f}%)")
        emb_issues = True
    if stats['chunks_with_excessive_newlines'] > 0:
        print(f"  ⚠️  Excessive newlines: {stats['chunks_with_excessive_newlines']:,} ({stats['chunks_with_excessive_newlines']/total*100:.2f}%)")
        emb_issues = True
    if stats['chunks_with_excessive_spaces'] > 0:
        print(f"  ⚠️  Excessive spaces: {stats['chunks_with_excessive_spaces']:,} ({stats['chunks_with_excessive_spaces']/total*100:.2f}%)")
        emb_issues = True
    if stats['chunks_with_urls_only'] > 0:
        print(f"  ⚠️  URL-only chunks: {stats['chunks_with_urls_only']:,} ({stats['chunks_with_urls_only']/total*100:.2f}%)")
        emb_issues = True
    if stats['chunks_with_citations_only'] > 0:
        print(f"  ⚠️  Citation-only chunks: {stats['chunks_with_citations_only']:,} ({stats['chunks_with_citations_only']/total*100:.2f}%)")
        emb_issues = True
    if not emb_issues:
        print("  ✅ No embedding-specific issues found")
    print()
    
    # Context and metadata
    print("CONTEXT & METADATA:")
    print(f"  Without context headers: {stats['chunks_without_context_headers']:,} ({stats['chunks_without_context_headers']/total*100:.2f}%)")
    print(f"  Without section info: {stats['chunks_without_section']:,} ({stats['chunks_without_section']/total*100:.2f}%)")
    print(f"  Without paper_id: {stats['chunks_without_paper_id']:,} ({stats['chunks_without_paper_id']/total*100:.2f}%)")
    print(f"  Without quality score: {stats['chunks_without_quality']:,} ({stats['chunks_without_quality']/total*100:.2f}%)")
    print(f"  Without entities: {stats['chunks_without_entities']:,} ({stats['chunks_without_entities']/total*100:.2f}%)")
    print(f"  Without keywords: {stats['chunks_without_keywords']:,} ({stats['chunks_without_keywords']/total*100:.2f}%)")
    print()
    
    # Consistency
    print("CONSISTENCY ISSUES:")
    if stats['chunks_length_mismatch'] > 0:
        print(f"  ⚠️  Length mismatches: {stats['chunks_length_mismatch']:,} ({stats['chunks_length_mismatch']/total*100:.2f}%)")
    if stats['chunks_metadata_incomplete'] > 0:
        print(f"  ⚠️  Incomplete metadata: {stats['chunks_metadata_incomplete']:,} ({stats['chunks_metadata_incomplete']/total*100:.2f}%)")
    if stats['chunks_length_mismatch'] == 0 and stats['chunks_metadata_incomplete'] == 0:
        print("  ✅ No consistency issues")
    print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS FOR EMBEDDING MODELS")
    print("=" * 80)
    print()
    
    recommendations = []
    
    if stats['chunks_too_small'] > 0:
        recommendations.append(f"⚠️  Filter {stats['chunks_too_small']:,} chunks that are too small (<50 chars) - may produce poor embeddings")
    
    if stats['chunks_too_large'] > 0:
        recommendations.append(f"⚠️  Split {stats['chunks_too_large']:,} chunks that are too large (>2000 chars) - may exceed model limits")
    
    if stats['chunks_token_count_none'] > 0:
        recommendations.append(f"⚠️  Calculate token counts for {stats['chunks_token_count_none']:,} chunks - needed for token-based models")
    
    if stats['chunks_over_token_limit'] > 0:
        recommendations.append(f"⚠️  {stats['chunks_over_token_limit']:,} chunks exceed 8192 token limit - split for compatibility")
    
    if stats['chunks_empty'] > 0 or stats['chunks_whitespace_only'] > 0:
        recommendations.append(f"⚠️  Remove {stats['chunks_empty'] + stats['chunks_whitespace_only']:,} empty/whitespace chunks - produce zero embeddings")
    
    if stats['chunks_with_html_tags'] > 0:
        recommendations.append(f"⚠️  Clean HTML tags from {stats['chunks_with_html_tags']:,} chunks - may affect embedding quality")
    
    if stats['chunks_with_excessive_newlines'] > 0:
        recommendations.append(f"⚠️  Normalize newlines in {stats['chunks_with_excessive_newlines']:,} chunks - improve consistency")
    
    if stats['chunks_without_context_headers'] > 0:
        recommendations.append(f"⚠️  Add context headers to {stats['chunks_without_context_headers']:,} chunks - improves retrieval")
    
    if stats['chunks_token_count_none'] > 0:
        recommendations.append(f"⚠️  Calculate token_count for {stats['chunks_token_count_none']:,} chunks - essential for token-based models")
    
    # Positive findings
    if stats['chunks_optimal_size'] / total > 0.8:
        recommendations.append(f"✅ Excellent size distribution: {stats['chunks_optimal_size']/total*100:.1f}% in optimal range")
    
    if stats['chunks_without_quality'] == 0:
        recommendations.append("✅ All chunks have quality scores - good for filtering")
    
    if stats['chunks_without_paper_id'] == 0:
        recommendations.append("✅ All chunks have paper_id - good for source attribution")
    
    if not recommendations:
        recommendations.append("✅ Chunks are well-optimized for embedding models!")
    
    for rec in recommendations:
        print(rec)
    
    print()
    print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze chunks for embedding model optimization')
    parser.add_argument('input_dir', type=str, help='Input directory (output_improved)')
    parser.add_argument('--sample', type=int, default=None, help='Sample size (for testing)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found")
        return
    
    stats = analyze_for_embeddings(input_dir, sample_size=args.sample)
    print_embedding_analysis(stats)
    
    # Save report
    report_file = Path('embedding_optimization_report.txt')
    import sys
    original_stdout = sys.stdout
    with open(report_file, 'w') as f:
        sys.stdout = f
        print_embedding_analysis(stats)
    sys.stdout = original_stdout
    
    print(f"\n✅ Detailed report saved to: {report_file}")

if __name__ == '__main__':
    main()
