#!/usr/bin/env python3
"""
Deduplicate arXiv papers by keeping only one copy per unique paper ID.
For papers that appear in multiple categories (cross-listed), keeps the copy
in the first category alphabetically and removes others.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def find_all_pdfs(pdfs_dir):
    """Find all PDF files and group by paper_id"""
    pdfs_dir = Path(pdfs_dir)
    papers = defaultdict(list)  # paper_id -> list of (category, path, mtime)
    
    print("Scanning PDF files...")
    count = 0
    for pdf_file in pdfs_dir.rglob("*.pdf"):
        if pdf_file.name.startswith("._"):
            continue  # Skip macOS resource forks
        
        # Extract category and paper_id from path: pdfs/cs.AI/2505/2505.01754.pdf
        parts = pdf_file.relative_to(pdfs_dir).parts
        if len(parts) >= 3:
            category = parts[0]
            paper_id = pdf_file.stem
            mtime = pdf_file.stat().st_mtime
            
            papers[paper_id].append((category, pdf_file, mtime))
            count += 1
            if count % 10000 == 0:
                print(f"  Scanned {count:,} files...", end='\r')
                sys.stdout.flush()
    
    print(f"\nFound {count:,} PDF files")
    return papers

def deduplicate_papers(papers, pdfs_dir, dry_run=True):
    """Remove duplicate papers, keeping one per paper_id"""
    pdfs_dir = Path(pdfs_dir)
    duplicates = {pid: paths for pid, paths in papers.items() if len(paths) > 1}
    
    print(f"\nFound {len(duplicates):,} papers with duplicates")
    print(f"Total duplicate files: {sum(len(paths) - 1 for paths in duplicates.values()):,}")
    
    if not duplicates:
        print("No duplicates found!")
        return
    
    # Calculate space that would be freed
    total_size = 0
    files_to_remove = []
    
    for paper_id, paths in duplicates.items():
        # Sort by category (alphabetically) to keep the first one
        paths_sorted = sorted(paths, key=lambda x: x[0])
        keep_path = paths_sorted[0][1]
        
        # Mark others for removal
        for category, path, mtime in paths_sorted[1:]:
            size = path.stat().st_size
            total_size += size
            files_to_remove.append((paper_id, category, path, size))
    
    print(f"\n{'Would free' if dry_run else 'Will free'}: {total_size / (1024**3):.2f} GB")
    print(f"Files to remove: {len(files_to_remove):,}")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be deleted ===")
        print("\nSample files that would be removed (first 20):")
        for paper_id, category, path, size in files_to_remove[:20]:
            print(f"  {category}/{path.name} ({size / 1024:.1f} KB)")
        if len(files_to_remove) > 20:
            print(f"  ... and {len(files_to_remove) - 20:,} more")
        print("\nRun with --execute to actually remove files")
        return
    
    # Actually remove files
    print("\nRemoving duplicate files...")
    removed = 0
    errors = 0
    
    for paper_id, category, path, size in files_to_remove:
        try:
            path.unlink()
            removed += 1
            if removed % 1000 == 0:
                print(f"  Removed {removed:,}/{len(files_to_remove):,} files...", end='\r')
                sys.stdout.flush()
        except Exception as e:
            errors += 1
            print(f"\nError removing {path}: {e}")
    
    print(f"\n\nRemoved {removed:,} duplicate files")
    if errors > 0:
        print(f"Errors: {errors}")
    print(f"Freed space: {total_size / (1024**3):.2f} GB")

def update_tracking_file(tracking_file, removed_files):
    """Update the successful downloads tracking file to remove deleted entries"""
    if not Path(tracking_file).exists():
        return
    
    print(f"\nUpdating tracking file: {tracking_file}")
    removed_paths = {str(f[2]) for f in removed_files}  # Set of paths to remove
    
    temp_file = tracking_file + '.tmp'
    kept = 0
    removed = 0
    
    with open(tracking_file, 'r') as infile, open(temp_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                # Check if this entry matches a removed file
                path = data.get('path', '')
                full_path = f"pdfs/{path}"
                
                if full_path in removed_paths:
                    removed += 1
                else:
                    outfile.write(line)
                    kept += 1
            except json.JSONDecodeError:
                # Keep malformed lines
                outfile.write(line)
                kept += 1
    
    os.replace(temp_file, tracking_file)
    print(f"  Kept {kept:,} entries, removed {removed:,} entries")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deduplicate arXiv papers by keeping one copy per unique paper ID',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--pdfs-dir', default='pdfs',
                       help='Directory containing PDF files (default: pdfs)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually remove files (default: dry run)')
    parser.add_argument('--update-tracking', action='store_true',
                       help='Update _successful_downloads.jsonl to remove deleted entries')
    
    args = parser.parse_args()
    
    pdfs_dir = Path(args.pdfs_dir)
    if not pdfs_dir.exists():
        print(f"Error: PDFs directory not found: {pdfs_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("arXiv Paper Deduplication Tool")
    print("=" * 70)
    print(f"PDFs directory: {pdfs_dir}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print()
    
    # Find all PDFs
    papers = find_all_pdfs(pdfs_dir)
    
    # Deduplicate
    files_to_remove = []
    duplicates = {pid: paths for pid, paths in papers.items() if len(paths) > 1}
    
    if duplicates:
        for paper_id, paths in duplicates.items():
            paths_sorted = sorted(paths, key=lambda x: x[0])
            for category, path, mtime in paths_sorted[1:]:
                size = path.stat().st_size
                files_to_remove.append((paper_id, category, path, size))
    
    deduplicate_papers(papers, pdfs_dir, dry_run=not args.execute)
    
    # Update tracking file if requested
    if args.execute and args.update_tracking and files_to_remove:
        tracking_file = pdfs_dir / '_successful_downloads.jsonl'
        if tracking_file.exists():
            update_tracking_file(str(tracking_file), files_to_remove)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
