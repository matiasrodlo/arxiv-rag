#!/usr/bin/env python3
"""
Fix embedding optimization issues:
1. Filter small chunks (<50 chars)
2. Clean HTML tags
3. Add missing context headers
4. Normalize excessive newlines
5. Split large chunks (>2000 chars)
"""

import json
import re
import html
from pathlib import Path
from typing import Dict, List
import argparse
import multiprocessing as mp
from functools import partial

def clean_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    try:
        text = html.unescape(text)
    except:
        pass
    return text

def normalize_newlines(text: str) -> str:
    """Normalize excessive newlines and line endings."""
    # Replace multiple newlines (3+) with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    return text

def split_text_intelligently(text: str, max_size: int = 2000) -> List[str]:
    """Split text intelligently at sentence or paragraph boundaries."""
    if len(text) <= max_size:
        return [text]
    
    parts = []
    
    # Try to split by paragraphs first
    paragraphs = text.split('\n\n')
    current_part = ""
    
    for para in paragraphs:
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
    part_start_in_original = original_text.find(part_text[:50]) if len(part_text) > 50 else 0
    if part_start_in_original == -1:
        part_start_in_original = 0
    
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
    
    return {
        'chunk_id': f"{new_meta.get('paper_id', 'unknown')}_chunk_{new_meta['chunk_index']}",
        'text': part_text,
        'metadata': new_meta
    }

def add_context_header(chunk: Dict, paper_title: str, section: str) -> Dict:
    """Add context header to chunk if missing."""
    text = chunk.get('text', '')
    
    # Check if already has header
    if text.startswith('#') or '##' in text[:100]:
        return chunk
    
    # Add header
    header = f"# {paper_title}\n\n## {section}\n\n"
    chunk['text'] = header + text
    
    # Update metadata
    meta = chunk.get('metadata', {})
    meta['chunk_length'] = len(chunk['text'])
    meta['has_context_header'] = True
    
    return chunk

def process_file(file_path: Path,
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 2000,
                 clean_html: bool = True,
                 normalize_newlines_flag: bool = True,
                 add_context: bool = True,
                 split_large: bool = True,
                 dry_run: bool = False) -> Dict:
    """Process a single file to fix embedding issues."""
    
    stats = {
        'file': str(file_path),
        'original_chunks': 0,
        'filtered_small': 0,
        'cleaned_html': 0,
        'normalized_newlines': 0,
        'added_context': 0,
        'split_large': 0,
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
        
        # Get paper metadata for context headers
        paper_metadata = data.get('metadata', {})
        paper_title = paper_metadata.get('title', 'Unknown Title')
        
        new_chunks = []
        
        for chunk in original_chunks:
            text = chunk.get('text', '')
            meta = chunk.get('metadata', {})
            
            # Filter small chunks
            if len(text) < min_chunk_size:
                stats['filtered_small'] += 1
                continue
            
            # Clean HTML tags
            if clean_html and re.search(r'<[^>]+>', text):
                text = clean_html_tags(text)
                stats['cleaned_html'] += 1
            
            # Normalize newlines
            if normalize_newlines_flag and text.count('\n') > len(text) / 10:
                text = normalize_newlines(text)
                stats['normalized_newlines'] += 1
            
            # Add context header if missing
            if add_context and not (text.startswith('#') or '##' in text[:100]):
                section = meta.get('section', 'Unknown Section')
                chunk = add_context_header(chunk, paper_title, section)
                text = chunk['text']
                stats['added_context'] += 1
            
            # Handle very long chunks
            if len(text) > max_chunk_size and split_large:
                parts = split_text_intelligently(text, max_size=max_chunk_size)
                
                if len(parts) > 1:
                    stats['split_large'] += 1
                    stats['chunks_after_split'] += len(parts)
                    
                    # Create new chunks from parts
                    for i, part in enumerate(parts):
                        new_chunk = create_chunk_from_part(part, chunk, i, len(parts))
                        new_chunks.append(new_chunk)
                else:
                    # Couldn't split intelligently, keep original
                    chunk['text'] = text
                    new_chunks.append(chunk)
            else:
                # Update chunk with cleaned text
                chunk['text'] = text
                meta['chunk_length'] = len(text)
                new_chunks.append(chunk)
        
        stats['final_chunks'] = len(new_chunks)
        
        # Update data
        if not dry_run:
            data['chunks'] = new_chunks
            
            # Update statistics if present
            if 'statistics' in data:
                data['statistics']['num_chunks'] = len(new_chunks)
                data['statistics']['embedding_optimized'] = True
            
            # Save updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        stats['errors'].append(str(e))
    
    return stats

def _process_file_worker(file_path: Path,
                         min_chunk_size: int,
                         max_chunk_size: int,
                         clean_html: bool,
                         normalize_newlines_flag: bool,
                         add_context: bool,
                         split_large: bool,
                         dry_run: bool) -> Dict:
    """Worker function for parallel processing."""
    return process_file(
        file_path,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        clean_html=clean_html,
        normalize_newlines_flag=normalize_newlines_flag,
        add_context=add_context,
        split_large=split_large,
        dry_run=dry_run
    )

def process_directory(input_dir: Path,
                     min_chunk_size: int = 50,
                     max_chunk_size: int = 2000,
                     clean_html: bool = True,
                     normalize_newlines_flag: bool = True,
                     add_context: bool = True,
                     split_large: bool = True,
                     dry_run: bool = False,
                     max_files: int = None,
                     num_workers: int = None,
                     verbose: bool = True):
    """Process all files in directory."""
    
    all_files = list(input_dir.rglob('*.json'))
    all_files = [f for f in all_files if not f.name.startswith('._')]
    
    if max_files:
        all_files = all_files[:max_files]
    
    total = len(all_files)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    if verbose:
        print("=" * 80)
        print("FIXING EMBEDDING OPTIMIZATION ISSUES")
        print("=" * 80)
        print()
        print(f"Processing {total:,} files...")
        print(f"  - Filtering chunks < {min_chunk_size} chars")
        print(f"  - Cleaning HTML tags: {clean_html}")
        print(f"  - Normalizing newlines: {normalize_newlines_flag}")
        print(f"  - Adding context headers: {add_context}")
        print(f"  - Splitting chunks > {max_chunk_size} chars: {split_large}")
        print(f"  - Mode: {'DRY RUN' if dry_run else 'WRITE'}")
        print(f"  - Workers: {num_workers}")
        print()
    
    total_stats = {
        'files_processed': 0,
        'files_with_errors': 0,
        'total_original_chunks': 0,
        'total_filtered_small': 0,
        'total_cleaned_html': 0,
        'total_normalized_newlines': 0,
        'total_added_context': 0,
        'total_split_large': 0,
        'total_after_split': 0,
        'total_final_chunks': 0,
        'errors': []
    }
    
    if num_workers == 1 or total == 1:
        # Sequential processing
        for i, file_path in enumerate(all_files, 1):
            if verbose and i % 10000 == 0:
                print(f"  Processed {i:,}/{total:,} files...")
            
            stats = process_file(
                file_path,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                clean_html=clean_html,
                normalize_newlines_flag=normalize_newlines_flag,
                add_context=add_context,
                split_large=split_large,
                dry_run=dry_run
            )
            
            total_stats['files_processed'] += 1
            total_stats['total_original_chunks'] += stats['original_chunks']
            total_stats['total_filtered_small'] += stats['filtered_small']
            total_stats['total_cleaned_html'] += stats['cleaned_html']
            total_stats['total_normalized_newlines'] += stats['normalized_newlines']
            total_stats['total_added_context'] += stats['added_context']
            total_stats['total_split_large'] += stats['split_large']
            total_stats['total_after_split'] += stats['chunks_after_split']
            total_stats['total_final_chunks'] += stats['final_chunks']
            
            if stats['errors']:
                total_stats['files_with_errors'] += 1
                total_stats['errors'].extend(stats['errors'])
    else:
        # Parallel processing
        worker_func = partial(
            _process_file_worker,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            clean_html=clean_html,
            normalize_newlines_flag=normalize_newlines_flag,
            add_context=add_context,
            split_large=split_large,
            dry_run=dry_run
        )
        
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for i, stats in enumerate(pool.imap_unordered(worker_func, all_files), 1):
                results.append(stats)
                if verbose and i % 10000 == 0:
                    print(f"  Processed {i:,}/{total:,} files...")
            
            # Aggregate statistics
            for stats in results:
                total_stats['files_processed'] += 1
                total_stats['total_original_chunks'] += stats.get('original_chunks', 0)
                total_stats['total_filtered_small'] += stats.get('filtered_small', 0)
                total_stats['total_cleaned_html'] += stats.get('cleaned_html', 0)
                total_stats['total_normalized_newlines'] += stats.get('normalized_newlines', 0)
                total_stats['total_added_context'] += stats.get('added_context', 0)
                total_stats['total_split_large'] += stats.get('split_large', 0)
                total_stats['total_after_split'] += stats.get('chunks_after_split', 0)
                total_stats['total_final_chunks'] += stats.get('final_chunks', 0)
                
                if stats.get('errors'):
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
        print("FIXES APPLIED:")
        print(f"  Filtered small chunks: {total_stats['total_filtered_small']:,}")
        print(f"  Cleaned HTML tags: {total_stats['total_cleaned_html']:,}")
        print(f"  Normalized newlines: {total_stats['total_normalized_newlines']:,}")
        print(f"  Added context headers: {total_stats['total_added_context']:,}")
        print(f"  Split large chunks: {total_stats['total_split_large']:,} → {total_stats['total_after_split']:,} chunks")
        print()
        print("CHUNK STATISTICS:")
        print(f"  Original chunks: {total_stats['total_original_chunks']:,}")
        print(f"  Final chunks: {total_stats['total_final_chunks']:,}")
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
    parser = argparse.ArgumentParser(description='Fix embedding optimization issues')
    parser.add_argument('input_dir', type=str, help='Input directory (output_improved)')
    parser.add_argument('--min-size', type=int, default=50,
                       help='Minimum chunk size in chars (default: 50)')
    parser.add_argument('--max-size', type=int, default=2000,
                       help='Maximum chunk size in chars (default: 2000)')
    parser.add_argument('--no-clean-html', action='store_true',
                       help='Do not clean HTML tags')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Do not normalize newlines')
    parser.add_argument('--no-context', action='store_true',
                       help='Do not add context headers')
    parser.add_argument('--no-split', action='store_true',
                       help='Do not split large chunks')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (do not write changes)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum files to process (for testing)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found")
        return
    
    process_directory(
        input_dir,
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size,
        clean_html=not args.no_clean_html,
        normalize_newlines_flag=not args.no_normalize,
        add_context=not args.no_context,
        split_large=not args.no_split,
        dry_run=args.dry_run,
        max_files=args.max_files,
        num_workers=args.workers,
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()
