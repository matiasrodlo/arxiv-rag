#!/usr/bin/env python3
"""
Generate paper IDs file from PDFs in the papers folder.
"""

import argparse
from pathlib import Path


def generate_paper_ids(pdf_dir: str = "papers", output_file: str = "paper_ids.txt"):
    """
    Generate a list of paper IDs from PDF files.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_file: Output file path for paper IDs
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    # Filter out system files (._*)
    pdf_files = [f for f in pdf_files if not f.name.startswith("._")]
    
    # Extract paper IDs (remove .pdf extension)
    paper_ids = sorted([f.stem for f in pdf_files])
    
    print(f"Found {len(paper_ids)} PDF files")
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for paper_id in paper_ids:
            f.write(f"{paper_id}\n")
    
    print(f"Paper IDs written to: {output_file}")
    print(f"First 5 IDs: {paper_ids[:5]}")
    print(f"Last 5 IDs: {paper_ids[-5:]}")
    
    return paper_ids


def main():
    parser = argparse.ArgumentParser(description="Generate paper IDs file from PDFs")
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='papers',
        help='Directory containing PDF files (default: papers)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='paper_ids.txt',
        help='Output file for paper IDs (default: paper_ids.txt)'
    )
    
    args = parser.parse_args()
    
    print("Generating paper IDs file...")
    generate_paper_ids(args.pdf_dir, args.output)


if __name__ == "__main__":
    main()

