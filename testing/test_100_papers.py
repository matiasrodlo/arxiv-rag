#!/usr/bin/env python3
"""
Test script to process 100 random papers and evaluate JSON output quality for RAG training.
"""

import sys
import json
import random
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from loguru import logger
import yaml

# Debug logging
DEBUG_LOG_PATH = Path("/Volumes/8SSD/ArxivCS/.cursor/debug.log")

def debug_log(location, message, data, hypothesis_id=None):
    """Write debug log entry."""
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.pdf_extractor import PDFExtractor, load_metadata
from src.processors.text_processor import TextProcessor, TextChunker
import yaml


def evaluate_json_quality(json_path: Path) -> Dict:
    """
    Evaluate the quality of a JSON file for RAG training.
    
    Returns a dictionary with quality metrics.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {
            'paper_id': json_path.stem,
            'error': str(e),
            'score': 0.0,
            'valid': False
        }
    
    issues = []
    warnings = []
    score = 10.0  # Start with perfect score
    
    # Check required fields
    required_fields = ['paper_id', 'metadata', 'text', 'chunks']
    for field in required_fields:
        if field not in data:
            issues.append(f"Missing required field: {field}")
            score -= 2.0
    
    if 'error' in locals():
        return {
            'paper_id': json_path.stem,
            'error': 'Missing required fields',
            'score': max(0.0, score),
            'valid': False
        }
    
    # Check metadata
    metadata = data.get('metadata', {})
    if not metadata.get('quality_score'):
        warnings.append("No quality score in metadata")
        score -= 0.5
    
    # Check text structure
    text_data = data.get('text', {})
    if not text_data.get('full'):
        issues.append("Missing full text")
        score -= 1.0
    
    if not text_data.get('by_page'):
        warnings.append("Missing page-level text")
        score -= 0.5
    
    if not text_data.get('sections'):
        warnings.append("Missing sections")
        score -= 0.5
    
    # Check chunks
    chunks = data.get('chunks', [])
    if not chunks:
        issues.append("No chunks found")
        score -= 2.0
    else:
        # Check chunk quality
        chunk_lengths = [len(c.get('text', '')) for c in chunks]
        if not chunk_lengths:
            issues.append("Chunks have no text")
            score -= 1.0
        else:
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            min_length = min(chunk_lengths)
            max_length = max(chunk_lengths)
            
            # Check for reasonable chunk sizes
            if min_length < 50:
                warnings.append(f"Some chunks too short (min: {min_length})")
                score -= 0.3
            
            if max_length > 5000:
                warnings.append(f"Some chunks too long (max: {max_length})")
                score -= 0.3
            
            # Check chunk metadata
            chunks_with_metadata = sum(1 for c in chunks if c.get('metadata'))
            if chunks_with_metadata < len(chunks) * 0.9:
                warnings.append(f"Some chunks missing metadata ({chunks_with_metadata}/{len(chunks)})")
                score -= 0.5
            
            # Check section mapping
            chunks_with_section = sum(1 for c in chunks 
                                     if c.get('metadata', {}).get('section') and 
                                     c['metadata']['section'] != 'Unknown')
            section_coverage = chunks_with_section / len(chunks) if chunks else 0
            if section_coverage < 0.7:
                warnings.append(f"Low section coverage ({section_coverage:.1%})")
                score -= 0.5
    
    # Check statistics
    stats = data.get('statistics', {})
    if not stats:
        warnings.append("Missing statistics")
        score -= 0.2
    
    return {
        'paper_id': data.get('paper_id', json_path.stem),
        'score': max(0.0, min(10.0, score)),
        'valid': score >= 7.0,
        'num_chunks': len(chunks),
        'num_sections': len(text_data.get('sections', [])),
        'num_pages': len(text_data.get('by_page', [])),
        'text_length': len(text_data.get('full', '')),
        'issues': issues,
        'warnings': warnings,
        'quality_score': metadata.get('quality_score', 0.0)
    }


def main():
    """Main test function."""
    # #region agent log
    debug_log("test_100_papers.py:160", "main() entry", {"script_started": True}, "A")
    # #endregion
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    # #region agent log
    debug_log("test_100_papers.py:163", "Loading config", {"config_path": str(config_path), "exists": config_path.exists()}, "A")
    # #endregion
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # #region agent log
    debug_log("test_100_papers.py:165", "Config loaded", {"has_pdf_dir": 'pdf_dir' in config.get('paths', {})}, "A")
    # #endregion
    
    # Setup paths
    papers_dir = Path(config['paths']['pdf_dir'])
    output_dir = Path(config['paths']['extracted_text_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of available papers
    pdf_files = list(papers_dir.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if not f.name.startswith("._")]
    
    if len(pdf_files) < 100:
        logger.warning(f"Only {len(pdf_files)} PDFs available, using all of them")
        selected_papers = [f.stem for f in pdf_files]
    else:
        # Randomly select 100 papers
        selected_papers = [f.stem for f in random.sample(pdf_files, 100)]
    
    logger.info(f"Selected {len(selected_papers)} papers for testing")
    
    # Initialize components (without embeddings/vector store)
    logger.info("Initializing components...")
    # #region agent log
    debug_log("test_100_papers.py:186", "Initializing PDF extractor", {}, "A")
    # #endregion
    
    pdf_extractor = PDFExtractor(
        enable_ocr=config['pdf_extraction']['enable_ocr'],
        ocr_language=config['pdf_extraction']['ocr_language']
    )
    
    # #region agent log
    debug_log("test_100_papers.py:190", "Initializing text processor", {}, "A")
    # #endregion
    
    text_processor = TextProcessor(
        remove_headers_footers=config['text_processing']['remove_headers_footers'],
        normalize_whitespace=config['text_processing']['normalize_whitespace'],
        fix_encoding=config['text_processing']['fix_encoding'],
        improve_formulas=config['text_processing'].get('improve_formulas', True)
    )
    
    # #region agent log
    debug_log("test_100_papers.py:197", "Text processor initialized", {}, "A")
    # #endregion
    
    # Force sentence chunking for testing (semantic requires sentence-transformers)
    chunking_method = config['chunking']['method']
    if chunking_method == "semantic":
        logger.warning("Semantic chunking not available for testing, using sentence chunking instead")
        chunking_method = "sentence"
    
    chunker = TextChunker(
        method=chunking_method,
        chunk_size=config['chunking']['chunk_size'],
        chunk_overlap=config['chunking']['chunk_overlap'],
        model_name=config['chunking'].get('model'),
        min_chunk_size=config['text_processing']['min_chunk_size'],
        max_chunk_size=config['text_processing']['max_chunk_size']
    )
    
    # #region agent log
    debug_log("test_100_papers.py:206", "Chunker initialized", {"method": chunker.method, "chunk_size": chunker.chunk_size, "has_model": chunker.model is not None}, "D")
    # #endregion
    
    # Process papers
    results = []
    failed = []
    
    logger.info("Processing papers...")
    # #region agent log
    debug_log("test_100_papers.py:191", "Starting paper processing loop", {"total_papers": len(selected_papers), "first_paper": selected_papers[0] if selected_papers else None}, "A")
    # #endregion
    
    for paper_idx, paper_id in enumerate(tqdm(selected_papers, desc="Processing")):
        try:
            # #region agent log
            debug_log("test_100_papers.py:194", "Processing paper", {"paper_idx": paper_idx, "paper_id": paper_id, "processed_count": len(results)}, "A")
            # #endregion
            
            # Extract PDF
            pdf_path = papers_dir / f"{paper_id}.pdf"
            if not pdf_path.exists():
                failed.append(paper_id)
                continue
            
            # #region agent log
            extract_start = time.time()
            debug_log("test_100_papers.py:200", "Starting PDF extraction", {"paper_id": paper_id, "pdf_path": str(pdf_path)}, "B")
            # #endregion
            
            extraction_result = pdf_extractor.extract(str(pdf_path))
            
            # #region agent log
            extract_time = time.time() - extract_start
            debug_log("test_100_papers.py:201", "PDF extraction completed", {"paper_id": paper_id, "extract_time": extract_time, "has_text": bool(extraction_result.get('text')), "text_length": len(extraction_result.get('text', ''))}, "B")
            # #endregion
            
            if not extraction_result or not extraction_result.get('success') or not extraction_result.get('text'):
                failed.append(paper_id)
                continue
            
            # #region agent log
            clean_start = time.time()
            debug_log("test_100_papers.py:206", "Starting text cleaning", {"paper_id": paper_id, "raw_text_length": len(extraction_result['text'])}, "C")
            # #endregion
            
            # Process text
            cleaned_text = text_processor.clean(extraction_result['text'])
            
            # #region agent log
            clean_time = time.time() - clean_start
            debug_log("test_100_papers.py:207", "Text cleaning completed", {"paper_id": paper_id, "clean_time": clean_time, "cleaned_length": len(cleaned_text)}, "C")
            # #endregion
            
            # #region agent log
            chunk_start = time.time()
            debug_log("test_100_papers.py:209", "Starting chunking", {"paper_id": paper_id, "text_length": len(cleaned_text), "chunking_method": chunker.method}, "D")
            # #endregion
            
            # Chunk text
            chunks = chunker.chunk(cleaned_text, metadata={'paper_id': paper_id})
            
            # #region agent log
            chunk_time = time.time() - chunk_start
            debug_log("test_100_papers.py:210", "Chunking completed", {"paper_id": paper_id, "chunk_time": chunk_time, "num_chunks": len(chunks)}, "D")
            # #endregion
            
            # #region agent log
            section_start = time.time()
            pages_data = extraction_result.get('pages', [])
            debug_log("test_100_papers.py:213", "Starting section extraction", {"paper_id": paper_id, "num_pages": len(pages_data)}, "E")
            # #endregion
            
            # Extract sections
            sections = text_processor.extract_sections(cleaned_text, pages=pages_data)
            
            # #region agent log
            section_time = time.time() - section_start
            debug_log("test_100_papers.py:214", "Section extraction completed", {"paper_id": paper_id, "section_time": section_time, "num_sections": len(sections)}, "E")
            # #endregion
            
            # Load metadata
            metadata = load_metadata(papers_dir / f"{paper_id}.txt") if (papers_dir / f"{paper_id}.txt").exists() else {}
            
            # Map chunks to sections
            chunk_start_positions = []
            current_pos = 0
            for chunk in chunks:
                chunk_start_positions.append(current_pos)
                current_pos += len(chunk['text']) + 1
            
            def find_section_for_position(pos):
                for s in sections:
                    if s['start_char'] <= pos < s['end_char']:
                        return s['name']
                return 'Unknown'
            
            def find_page_for_position(pos):
                current_char_count = 0
                for p in pages_data:
                    current_char_count += p.get('char_count', 0) + 2
                    if pos < current_char_count:
                        return p.get('page', 1)
                return 1
            
            # Create JSON structure
            json_data = {
                'paper_id': paper_id,
                'metadata': {
                    **metadata,
                    'extraction_method': extraction_result['method_used'],
                    'quality_score': extraction_result.get('quality_score', 0.0),
                    'pdf_type': extraction_result.get('pdf_type', 'unknown'),
                    'num_pages': len(pages_data),
                    'text_length': len(cleaned_text),
                },
                'text': {
                    'full': cleaned_text,
                    'by_page': [
                        {
                            'page': p.get('page', i + 1),
                            'text': p.get('text', ''),
                            'char_count': p.get('char_count', len(p.get('text', '')))
                        }
                        for i, p in enumerate(pages_data)
                    ],
                    'sections': [
                        {
                            'name': s.get('name', ''),
                            'text': s.get('text', ''),
                            'start_char': s.get('start_char', 0),
                            'end_char': s.get('end_char', 0),
                            'page': s.get('page', 1)
                        }
                        for s in sections
                    ]
                },
                'chunks': [
                    {
                        'chunk_id': f"{paper_id}_chunk_{i}",
                        'text': chunk['text'],
                        'metadata': {
                            **chunk.get('metadata', {}),
                            'chunk_index': i,
                            'chunk_length': len(chunk['text']),
                            'paper_id': paper_id,
                            'section': find_section_for_position(chunk_start_positions[i]) if i < len(chunk_start_positions) else 'Unknown',
                            'page': find_page_for_position(chunk_start_positions[i]) if i < len(chunk_start_positions) else 1
                        }
                    }
                    for i, chunk in enumerate(chunks)
                ],
                'statistics': {
                    'num_chunks': len(chunks),
                    'num_pages': len(pages_data),
                    'num_sections': len(sections),
                    'total_chars': len(cleaned_text),
                    'avg_chunk_size': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0,
                    'chunking_method': chunker.method
                }
            }
            
            # #region agent log
            save_start = time.time()
            debug_log("test_100_papers.py:297", "Starting JSON save", {"paper_id": paper_id, "json_size": len(json.dumps(json_data))}, "F")
            # #endregion
            
            # Save JSON
            json_path = output_dir / f"{paper_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # #region agent log
            save_time = time.time() - save_start
            debug_log("test_100_papers.py:299", "JSON save completed", {"paper_id": paper_id, "save_time": save_time, "json_path": str(json_path)}, "F")
            # #endregion
            
            results.append(paper_id)
            
            # #region agent log
            debug_log("test_100_papers.py:301", "Paper processing complete", {"paper_id": paper_id, "total_processed": len(results), "total_failed": len(failed)}, "A")
            # #endregion
            
        except Exception as e:
            # #region agent log
            debug_log("test_100_papers.py:303", "Exception in paper processing", {"paper_id": paper_id, "error": str(e), "error_type": type(e).__name__}, "G")
            # #endregion
            logger.error(f"Error processing {paper_id}: {e}")
            failed.append(paper_id)
    
    logger.info(f"Processed {len(results)} papers successfully, {len(failed)} failed")
    
    # Evaluate JSON quality
    logger.info("Evaluating JSON quality...")
    evaluations = []
    
    for paper_id in tqdm(results, desc="Evaluating"):
        json_path = output_dir / f"{paper_id}.json"
        if json_path.exists():
            eval_result = evaluate_json_quality(json_path)
            evaluations.append(eval_result)
        else:
            logger.warning(f"JSON file not found for {paper_id}")
    
    # Generate report
    if evaluations:
        avg_score = sum(e['score'] for e in evaluations) / len(evaluations)
        valid_count = sum(1 for e in evaluations if e['valid'])
        avg_chunks = sum(e['num_chunks'] for e in evaluations) / len(evaluations)
        avg_sections = sum(e['num_sections'] for e in evaluations) / len(evaluations)
        avg_text_length = sum(e['text_length'] for e in evaluations) / len(evaluations)
        avg_quality = sum(e['quality_score'] for e in evaluations) / len(evaluations)
        
        # Count issues
        total_issues = sum(len(e['issues']) for e in evaluations)
        total_warnings = sum(len(e['warnings']) for e in evaluations)
        
        # Generate report
        report = {
            'summary': {
                'total_papers': len(selected_papers),
                'processed': len(results),
                'failed': len(failed),
                'evaluated': len(evaluations),
                'valid': valid_count,
                'invalid': len(evaluations) - valid_count,
                'avg_score': round(avg_score, 2),
                'avg_chunks': round(avg_chunks, 1),
                'avg_sections': round(avg_sections, 1),
                'avg_text_length': round(avg_text_length, 0),
                'avg_quality_score': round(avg_quality, 3),
                'total_issues': total_issues,
                'total_warnings': total_warnings
            },
            'failed_papers': failed,
            'evaluations': evaluations
        }
        
        # Save report
        report_path = Path(__file__).parent / "test_100_papers_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Papers: {len(selected_papers)}")
        print(f"Successfully Processed: {len(results)}")
        print(f"Failed: {len(failed)}")
        print(f"Evaluated: {len(evaluations)}")
        print(f"Valid for RAG: {valid_count} ({valid_count/len(evaluations)*100:.1f}%)")
        print(f"Average Quality Score: {avg_score:.2f}/10.0")
        print(f"Average Chunks per Paper: {avg_chunks:.1f}")
        print(f"Average Sections per Paper: {avg_sections:.1f}")
        print(f"Average Text Length: {avg_text_length:,.0f} characters")
        print(f"Average Extraction Quality: {avg_quality:.3f}")
        print(f"Total Issues: {total_issues}")
        print(f"Total Warnings: {total_warnings}")
        print("=" * 80)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Show papers with issues
        papers_with_issues = [e for e in evaluations if e['issues'] or e['score'] < 7.0]
        if papers_with_issues:
            print(f"\n⚠️  {len(papers_with_issues)} papers with issues:")
            for e in sorted(papers_with_issues, key=lambda x: x['score'])[:10]:
                print(f"  • {e['paper_id']}: Score {e['score']:.1f} - {len(e['issues'])} issues, {len(e['warnings'])} warnings")
    else:
        logger.error("No evaluations completed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

