#!/usr/bin/env python3
"""
Fix chunk quality issues:
1. Filter low-quality chunks (<0.7 quality score)
2. Split very long chunks (>2000 chars) into smaller chunks
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import argparse
from collections import defaultdict

def split_text_intelligently(text: str, max_size: int = 1500) -> List[str]:
    """Split text intelligently at sentence or paragraph boundaries."""
    if len(text) <= max_size:
        return [text]
    
    parts = []
    
    # Try to split by paragraphs first
    paragraphs = text.split('\n\n')
    current_part = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed max_size
        if current_part and len(current_part) + len(para) + 2 > max_size:
            if current_part:
                parts.append(current_part.strip())
            current_part = para
        else:
            if current_part:
                current_part += "\n\n" + para
            else:
                current_part = para
    
    # If still too long, split by sentences
    if len(current_part) > max_size:
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', current_part)
        current_sent = ""
        
        for sent in sentences:
            if current_sent and len(current_sent) + len(sent) + 1 > max_size:
                if current_sent:
                    parts.append(current_sent.strip())
                current_sent = sent
            else:
                if current_sent:
                    current_sent += " " + sent
                else:
                    current_sent = sent
        
        if current_sent:
            parts.append(current_sent.strip())
    else:
        if current_part:
            parts.append(current_part.strip())
    
    return parts if parts else [text]

def create_chunk_from_part(part_text: str, original_chunk: Dict, part_index: int, total_parts: int) -> Dict:
    """Create a new chunk from a split part."""
    original_meta = original_chunk.get('metadata', {})
    
    # Calculate new char positions
    original_start = original_meta.get('char_start', 0)
    original_text = original_chunk.get('text', '')
    
    # Find position of this part in original text
    part_start_in_original = original_text.find(part_text[:50])  # Find by first 50 chars
    if part_start_in_original == -1:
        part_start_in_original = original_start
    
    new_char_start = original_start + part_start_in_original
    new_char_end = new_char_start + len(part_text)
    
    # Create new metadata
    new_meta = original_meta.copy()
    new_meta['chunk_length'] = len(part_text)
    new_meta['char_start'] = new_char_start
    new_meta['char_end'] = new_char_end
    
    # Update chunk index if this is a split
    if total_parts > 1:
        original_index = original_meta.get('chunk_index', 0)
        new_meta['chunk_index'] = f"{original_index}.{part_index}"
        new_meta['is_split'] = True
        new_meta['split_part'] = part_index
        new_meta['split_total'] = total_parts
        new_meta['original_chunk_index'] = original_index
    
    # Recalculate word count
    new_meta['word_count'] = len(part_text.split())
    
    # Extract citations from this part
    citations = new_meta.get('citations', [])
    if citations and isinstance(citations, list):
        # Filter citations that appear in this part
        part_citations = []
        for c in citations:
            try:
                cit_str = str(c) if not isinstance(c, str) else c
                if cit_str and (cit_str in part_text or part_text.find(cit_str[:20]) != -1):
                    part_citations.append(c)
            except:
                continue
        new_meta['citations'] = part_citations
        new_meta['citation_count'] = len(part_citations)
        new_meta['has_citations'] = len(part_citations) > 0
    else:
        # No citations or invalid format
        new_meta['citations'] = []
        new_meta['citation_count'] = 0
        new_meta['has_citations'] = False
    
    return {
        'chunk_id': f"{new_meta.get('paper_id', 'unknown')}_chunk_{new_meta['chunk_index']}",
        'text': part_text,
        'metadata': new_meta
    }

def process_file(file_path: Path, 
                 min_quality: float = 0.7,
                 max_chunk_size: int = 2000,
                 split_long_chunks: bool = True,
                 dry_run: bool = False) -> Dict:
    """Process a single file to fix chunk issues."""
    
    stats = {
        'file': str(file_path),
        'original_chunks': 0,
        'filtered_low_quality': 0,
        'split_chunks': 0,
        'chunks_after_split': 0,
        'final_chunks': 0,
        'errors': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_chunks = data.get('chunks', [])
        stats['original_chunks'] = len(original_chunks)
        
        if not original_chunks:
            return stats
        
        new_chunks = []
        
        for chunk in original_chunks:
            text = chunk.get('text', '')
            meta = chunk.get('metadata', {})
            quality_score = meta.get('quality_score', 1.0)
            
            # Filter low-quality chunks
            if quality_score < min_quality:
                stats['filtered_low_quality'] += 1
                continue
            
            # Handle very long chunks
            if len(text) > max_chunk_size and split_long_chunks:
                # Split the chunk
                parts = split_text_intelligently(text, max_size=max_chunk_size)
                
                if len(parts) > 1:
                    stats['split_chunks'] += 1
                    stats['chunks_after_split'] += len(parts)
                    
                    # Create new chunks from parts
                    for i, part in enumerate(parts):
                        new_chunk = create_chunk_from_part(part, chunk, i, len(parts))
                        new_chunks.append(new_chunk)
                else:
                    # Couldn't split intelligently, keep original
                    new_chunks.append(chunk)
            else:
                # Keep chunk as-is
                new_chunks.append(chunk)
        
        stats['final_chunks'] = len(new_chunks)
        
        # Update data
        if not dry_run:
            data['chunks'] = new_chunks
            
            # Update statistics if present
            if 'statistics' in data:
                data['statistics']['num_chunks'] = len(new_chunks)
                data['statistics']['chunks_filtered'] = stats['filtered_low_quality']
                data['statistics']['chunks_split'] = stats['split_chunks']
            
            # Save updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        stats['errors'].append(str(e))
    
    return stats

def process_directory(input_dir: Path,
                     min_quality: float = 0.7,
                     max_chunk_size: int = 2000,
                     split_long_chunks: bool = True,
                     dry_run: bool = False,
                     max_files: int = None,
                     verbose: bool = True):
    """Process all files in directory."""
    
    all_files = list(input_dir.rglob('*.json'))
    all_files = [f for f in all_files if not f.name.startswith('._')]
    
    if max_files:
        all_files = all_files[:max_files]
    
    total = len(all_files)
    
    if verbose:
        print("=" * 80)
        print("FIXING CHUNK ISSUES")
        print("=" * 80)
        print()
        print(f"Processing {total:,} files...")
        print(f"  - Filtering chunks with quality < {min_quality}")
        print(f"  - Splitting chunks > {max_chunk_size} chars: {split_long_chunks}")
        print(f"  - Mode: {'DRY RUN' if dry_run else 'WRITE'}")
        print()
    
    total_stats = {
        'files_processed': 0,
        'files_with_errors': 0,
        'total_original_chunks': 0,
        'total_filtered': 0,
        'total_split': 0,
        'total_after_split': 0,
        'total_final_chunks': 0,
        'errors': []
    }
    
    for i, file_path in enumerate(all_files, 1):
        if verbose and i % 1000 == 0:
            print(f"  Processed {i:,}/{total:,} files...")
        
        stats = process_file(
            file_path,
            min_quality=min_quality,
            max_chunk_size=max_chunk_size,
            split_long_chunks=split_long_chunks,
            dry_run=dry_run
        )
        
        total_stats['files_processed'] += 1
        total_stats['total_original_chunks'] += stats['original_chunks']
        total_stats['total_filtered'] += stats['filtered_low_quality']
        total_stats['total_split'] += stats['split_chunks']
        total_stats['total_after_split'] += stats['chunks_after_split']
        total_stats['total_final_chunks'] += stats['final_chunks']
        
        if stats['errors']:
            total_stats['files_with_errors'] += 1
            total_stats['errors'].extend(stats['errors'])
    
    # Print summary
    if verbose:
        print()
        print("=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print()
        print(f"Files processed: {total_stats['files_processed']:,}")
        print(f"Files with errors: {total_stats['files_with_errors']:,}")
        print()
        print("CHUNK STATISTICS:")
        print(f"  Original chunks: {total_stats['total_original_chunks']:,}")
        print(f"  Filtered (low quality): {total_stats['total_filtered']:,}")
        print(f"  Split (very long): {total_stats['total_split']:,}")
        print(f"  Chunks after splitting: {total_stats['total_after_split']:,}")
        print(f"  Final chunks: {total_stats['total_final_chunks']:,}")
        print()
        
        if total_stats['total_original_chunks'] > 0:
            reduction = total_stats['total_filtered']
            reduction_pct = (reduction / total_stats['total_original_chunks']) * 100
            print(f"  Low-quality chunks removed: {reduction:,} ({reduction_pct:.3f}%)")
        
        if total_stats['total_split'] > 0:
            split_increase = total_stats['total_after_split'] - total_stats['total_split']
            print(f"  Long chunks split: {total_stats['total_split']:,} → {total_stats['total_after_split']:,} chunks (+{split_increase:,})")
        
        net_change = total_stats['total_final_chunks'] - total_stats['total_original_chunks']
        print(f"  Net change: {net_change:+,} chunks")
        print()
        
        if total_stats['errors']:
            print(f"⚠️  Errors encountered: {len(total_stats['errors'])}")
            for error in total_stats['errors'][:10]:
                print(f"  {error}")
            if len(total_stats['errors']) > 10:
                print(f"  ... and {len(total_stats['errors']) - 10} more errors")
        
        print("=" * 80)
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(description='Fix chunk quality issues')
    parser.add_argument('input_dir', type=str, help='Input directory (output_improved)')
    parser.add_argument('--min-quality', type=float, default=0.7, 
                       help='Minimum quality score (default: 0.7)')
    parser.add_argument('--max-chunk-size', type=int, default=2000,
                       help='Maximum chunk size in chars (default: 2000)')
    parser.add_argument('--no-split', action='store_true',
                       help='Do not split long chunks, only filter low quality')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (do not write changes)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum files to process (for testing)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found")
        return
    
    process_directory(
        input_dir,
        min_quality=args.min_quality,
        max_chunk_size=args.max_chunk_size,
        split_long_chunks=not args.no_split,
        dry_run=args.dry_run,
        max_files=args.max_files,
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()
