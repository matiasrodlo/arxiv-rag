#!/usr/bin/env python3
"""
Comprehensive JSON Quality Analysis for AI Scientist RAG
Analyzes extracted paper JSON files to assess their suitability for RAG-powered autonomous research.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import statistics
import re

class JSONQualityAnalyzer:
    """Analyze JSON files for RAG quality and completeness."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results = []
        self.errors = []
        
    def analyze_file(self, json_path: Path) -> Dict:
        """Analyze a single JSON file."""
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
        score = 10.0  # Start with perfect score
        
        paper_id = data.get('paper_id', json_path.stem)
        
        # 1. Check required structure (critical for RAG)
        required_fields = ['paper_id', 'metadata', 'text', 'chunks']
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing critical field: {field}")
                score -= 2.0
        
        # 2. Evaluate metadata quality
        metadata = data.get('metadata', {})
        metadata_score = 0
        if metadata.get('title'):
            metadata_score += 1
            strengths.append("Has title")
        else:
            warnings.append("Missing title")
        
        if metadata.get('authors'):
            metadata_score += 1
            strengths.append("Has authors")
        else:
            warnings.append("Missing authors")
        
        if metadata.get('abstract'):
            metadata_score += 1
            strengths.append("Has abstract")
        else:
            warnings.append("Missing abstract")
        
        extraction_method = metadata.get('extraction_method', 'unknown')
        quality_score = metadata.get('quality_score', 0.0)
        
        # 3. Evaluate text quality
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
        
        # Check for common extraction artifacts
        artifact_patterns = [
            (r'\[.*?\]', 'Brackets/References'),
            (r'Figure \d+', 'Figure references'),
            (r'Table \d+', 'Table references'),
            (r'Equation \d+', 'Equation references'),
        ]
        
        text_quality_issues = []
        for pattern, name in artifact_patterns:
            matches = len(re.findall(pattern, full_text))
            if matches > 100:  # Too many might indicate poor extraction
                text_quality_issues.append(f"Many {name}: {matches}")
        
        # 4. Evaluate sections (important for structured retrieval)
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
        section_names = [s.get('name', '') for s in sections]
        common_sections = ['Introduction', 'Abstract', 'Method', 'Results', 'Conclusion', 'Related Work']
        found_sections = sum(1 for name in section_names if any(common in name for common in common_sections))
        if found_sections < 2:
            warnings.append(f"Missing common sections (found {found_sections})")
        
        # 5. Evaluate chunks (CRITICAL for RAG)
        chunks = data.get('chunks', [])
        num_chunks = len(chunks)
        
        if num_chunks == 0:
            issues.append("No chunks - cannot build RAG")
            score -= 5.0
        elif num_chunks < 5:
            issues.append(f"Too few chunks: {num_chunks}")
            score -= 2.0
        else:
            strengths.append(f"Good chunk count: {num_chunks} chunks")
        
        # Analyze chunk quality
        chunk_lengths = []
        chunks_with_metadata = 0
        chunks_with_section = 0
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_lengths.append(len(chunk_text))
            
            chunk_meta = chunk.get('metadata', {})
            if chunk_meta:
                chunks_with_metadata += 1
                if chunk_meta.get('section'):
                    chunks_with_section += 1
        
        if chunk_lengths:
            avg_chunk_length = statistics.mean(chunk_lengths)
            min_chunk_length = min(chunk_lengths)
            max_chunk_length = max(chunk_lengths)
            
            # Ideal chunk size: 200-1000 chars for RAG
            if avg_chunk_length < 100:
                issues.append(f"Chunks too short (avg: {avg_chunk_length:.0f} chars)")
                score -= 1.0
            elif avg_chunk_length > 2000:
                warnings.append(f"Chunks quite long (avg: {avg_chunk_length:.0f} chars)")
                score -= 0.3
            else:
                strengths.append(f"Good chunk size (avg: {avg_chunk_length:.0f} chars)")
            
            if min_chunk_length < 50:
                warnings.append(f"Some chunks very short (min: {min_chunk_length} chars)")
        
        metadata_coverage = chunks_with_metadata / num_chunks if num_chunks > 0 else 0
        section_coverage = chunks_with_section / num_chunks if num_chunks > 0 else 0
        
        if metadata_coverage < 0.8:
            warnings.append(f"Low metadata coverage: {metadata_coverage:.1%}")
        else:
            strengths.append(f"Good metadata coverage: {metadata_coverage:.1%}")
        
        if section_coverage < 0.5:
            warnings.append(f"Low section coverage: {section_coverage:.1%}")
        else:
            strengths.append(f"Good section coverage: {section_coverage:.1%}")
        
        # 6. Evaluate statistics
        stats = data.get('statistics', {})
        if not stats:
            warnings.append("Missing statistics")
        else:
            strengths.append("Has statistics")
        
        # 7. RAG-specific quality checks
        rag_issues = []
        
        # Check for embeddings (if present)
        chunks_with_embeddings = sum(1 for chunk in chunks if 'embedding' in chunk.get('metadata', {}))
        if chunks_with_embeddings == 0:
            rag_issues.append("No embeddings in chunks (may need to generate)")
        
        # Check for semantic coherence
        # Look for very short or very repetitive chunks
        unique_chunks = len(set(chunk.get('text', '')[:100] for chunk in chunks))  # First 100 chars
        if unique_chunks < num_chunks * 0.8:
            warnings.append(f"Potential duplicate chunks: {unique_chunks}/{num_chunks} unique")
        
        # 8. Calculate final score
        score = max(0.0, min(10.0, score))  # Clamp between 0 and 10
        
        return {
            'paper_id': paper_id,
            'score': round(score, 2),
            'text_length': text_length,
            'num_chunks': num_chunks,
            'num_sections': num_sections,
            'extraction_method': extraction_method,
            'quality_score': quality_score,
            'avg_chunk_length': statistics.mean(chunk_lengths) if chunk_lengths else 0,
            'metadata_coverage': metadata_coverage,
            'section_coverage': section_coverage,
            'issues': issues,
            'warnings': warnings,
            'strengths': strengths,
            'rag_issues': rag_issues
        }
    
    def analyze_all(self, sample_size: Optional[int] = None) -> Dict:
        """Analyze all JSON files in the output directory."""
        json_files = list(self.output_dir.glob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith('._')]  # Filter macOS files
        
        if sample_size:
            import random
            json_files = random.sample(json_files, min(sample_size, len(json_files)))
        
        print(f"Analyzing {len(json_files)} JSON files...")
        
        for json_file in json_files:
            result = self.analyze_file(json_file)
            self.results.append(result)
            if result.get('error'):
                self.errors.append(result)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive quality report."""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        valid_results = [r for r in self.results if 'error' not in r]
        
        # Overall statistics
        scores = [r['score'] for r in valid_results]
        text_lengths = [r['text_length'] for r in valid_results]
        num_chunks = [r['num_chunks'] for r in valid_results]
        num_sections = [r['num_sections'] for r in valid_results]
        
        # Extraction method distribution
        extraction_methods = Counter(r['extraction_method'] for r in valid_results)
        
        # Issue analysis
        all_issues = []
        all_warnings = []
        all_strengths = []
        for r in valid_results:
            all_issues.extend(r.get('issues', []))
            all_warnings.extend(r.get('warnings', []))
            all_strengths.extend(r.get('strengths', []))
        
        issue_counts = Counter(all_issues)
        warning_counts = Counter(all_warnings)
        strength_counts = Counter(all_strengths)
        
        # Quality distribution
        excellent = sum(1 for s in scores if s >= 9.0)
        good = sum(1 for s in scores if 7.0 <= s < 9.0)
        fair = sum(1 for s in scores if 5.0 <= s < 7.0)
        poor = sum(1 for s in scores if s < 5.0)
        
        report = {
            'summary': {
                'total_files': len(self.results),
                'valid_files': len(valid_results),
                'error_files': len(self.errors),
                'avg_score': statistics.mean(scores) if scores else 0,
                'median_score': statistics.median(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
            },
            'text_quality': {
                'avg_length': statistics.mean(text_lengths) if text_lengths else 0,
                'median_length': statistics.median(text_lengths) if text_lengths else 0,
                'min_length': min(text_lengths) if text_lengths else 0,
                'max_length': max(text_lengths) if text_lengths else 0,
            },
            'chunk_quality': {
                'avg_chunks': statistics.mean(num_chunks) if num_chunks else 0,
                'median_chunks': statistics.median(num_chunks) if num_chunks else 0,
                'min_chunks': min(num_chunks) if num_chunks else 0,
                'max_chunks': max(num_chunks) if num_chunks else 0,
            },
            'section_quality': {
                'avg_sections': statistics.mean(num_sections) if num_sections else 0,
                'median_sections': statistics.median(num_sections) if num_sections else 0,
            },
            'extraction_methods': dict(extraction_methods),
            'quality_distribution': {
                'excellent (9-10)': excellent,
                'good (7-9)': good,
                'fair (5-7)': fair,
                'poor (<5)': poor,
            },
            'common_issues': dict(issue_counts.most_common(10)),
            'common_warnings': dict(warning_counts.most_common(10)),
            'common_strengths': dict(strength_counts.most_common(10)),
            'rag_readiness': {
                'files_with_chunks': sum(1 for r in valid_results if r['num_chunks'] > 0),
                'files_with_sections': sum(1 for r in valid_results if r['num_sections'] > 0),
                'files_ready_for_rag': sum(1 for r in valid_results if r['score'] >= 7.0 and r['num_chunks'] > 0),
            }
        }
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze JSON quality for RAG')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory containing JSON files')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for analysis (None = all files)')
    parser.add_argument('--output', type=str, default='json_quality_report.json',
                       help='Output file for report')
    
    args = parser.parse_args()
    
    analyzer = JSONQualityAnalyzer(Path(args.output_dir))
    report = analyzer.analyze_all(sample_size=args.sample_size)
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("JSON QUALITY ANALYSIS REPORT")
    print("="*80)
    print(f"\nTotal files analyzed: {report['summary']['total_files']}")
    print(f"Valid files: {report['summary']['valid_files']}")
    print(f"Error files: {report['summary']['error_files']}")
    print(f"\nAverage Quality Score: {report['summary']['avg_score']:.2f}/10.0")
    print(f"Median Quality Score: {report['summary']['median_score']:.2f}/10.0")
    print(f"\nQuality Distribution:")
    for category, count in report['quality_distribution'].items():
        pct = (count / report['summary']['valid_files'] * 100) if report['summary']['valid_files'] > 0 else 0
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    print(f"\nText Quality:")
    print(f"  Average length: {report['text_quality']['avg_length']:,.0f} chars")
    print(f"  Median length: {report['text_quality']['median_length']:,.0f} chars")
    
    print(f"\nChunk Quality:")
    print(f"  Average chunks per paper: {report['chunk_quality']['avg_chunks']:.1f}")
    print(f"  Median chunks per paper: {report['chunk_quality']['median_chunks']:.1f}")
    
    print(f"\nRAG Readiness:")
    print(f"  Files with chunks: {report['rag_readiness']['files_with_chunks']}")
    print(f"  Files with sections: {report['rag_readiness']['files_with_sections']}")
    print(f"  Files ready for RAG (score >= 7.0): {report['rag_readiness']['files_ready_for_rag']}")
    
    print(f"\nTop Issues:")
    for issue, count in list(report['common_issues'].items())[:5]:
        print(f"  {issue}: {count}")
    
    print(f"\nTop Strengths:")
    for strength, count in list(report['common_strengths'].items())[:5]:
        print(f"  {strength}: {count}")
    
    print(f"\nReport saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()

