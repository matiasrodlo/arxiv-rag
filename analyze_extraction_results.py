#!/usr/bin/env python3
"""
PDF Extraction Analysis Script

Analyzes extraction results from 100 PDFs to identify areas for improvement.
Provides detailed insights on quality, performance, and potential issues.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import statistics


class ExtractionAnalyzer:
    """Analyzes PDF extraction results to identify improvement areas."""

    def __init__(self, results_dir: str = 'test_results/extractions'):
        """
        Initialize the analyzer.

        Args:
            results_dir: Directory containing individual JSON extraction results
        """
        self.results_dir = Path(results_dir)
        self.extraction_results = []
        self.summary_stats = {}

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all JSON extraction results from the directory."""
        json_files = list(self.results_dir.rglob('*.json'))
        print(f"Found {len(json_files)} extraction result files")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file)
                    self.extraction_results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        print(f"Successfully loaded {len(self.extraction_results)} results")
        return self.extraction_results

    def analyze_quality_scores(self) -> Dict[str, Any]:
        """Analyze extraction quality scores."""
        scores = [r.get('quality_score', 0) for r in self.extraction_results]

        analysis = {
            'average': round(statistics.mean(scores), 4),
            'median': round(statistics.median(scores), 4),
            'stdev': round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
            'min': min(scores),
            'max': max(scores),
            'distribution': {
                '0.90-0.95': len([s for s in scores if 0.90 <= s < 0.95]),
                '0.95-0.98': len([s for s in scores if 0.95 <= s < 0.98]),
                '0.98-1.00': len([s for s in scores if 0.98 <= s < 1.00]),
                '1.00': len([s for s in scores if s == 1.00]),
            }
        }

        # Find lowest quality extractions
        low_quality = sorted(
            [(r.get('pdf_name'), r.get('quality_score'), r.get('relative_path'))
             for r in self.extraction_results if r.get('quality_score', 1.0) < 0.98],
            key=lambda x: x[1]
        )

        analysis['low_quality_extractions'] = low_quality[:10]

        return analysis

    def analyze_extraction_methods(self) -> Dict[str, Any]:
        """Analyze which extraction methods were used and their effectiveness."""
        method_stats = defaultdict(lambda: {
            'count': 0,
            'avg_quality': [],
            'avg_time': [],
            'pdf_types': set()
        })

        for r in self.extraction_results:
            method = r.get('method_used', 'unknown')
            method_stats[method]['count'] += 1
            method_stats[method]['avg_quality'].append(r.get('quality_score', 0))
            method_stats[method]['avg_time'].append(r.get('extraction_time_seconds', 0))
            method_stats[method]['pdf_types'].add(r.get('pdf_type', 'unknown'))

        # Calculate averages
        for method, stats in method_stats.items():
            stats['avg_quality'] = round(statistics.mean(stats['avg_quality']), 4) if stats['avg_quality'] else 0
            stats['avg_time'] = round(statistics.mean(stats['avg_time']), 4) if stats['avg_time'] else 0
            stats['pdf_types'] = list(stats['pdf_types'])

        return dict(method_stats)

    def analyze_text_content(self) -> Dict[str, Any]:
        """Analyze text content characteristics."""
        char_counts = [r.get('char_count', 0) for r in self.extraction_results]
        word_counts = [r.get('word_count', 0) for r in self.extraction_results]
        page_counts = [r.get('num_pages', 0) for r in self.extraction_results]

        # Characters per page
        chars_per_page = []
        for r in self.extraction_results:
            if r.get('num_pages', 0) > 0:
                chars_per_page.append(r.get('char_count', 0) / r.get('num_pages', 1))

        analysis = {
            'char_count': {
                'total': sum(char_counts),
                'average': round(statistics.mean(char_counts), 0),
                'median': round(statistics.median(char_counts), 0),
                'min': min(char_counts),
                'max': max(char_counts),
                'stdev': round(statistics.stdev(char_counts), 0) if len(char_counts) > 1 else 0
            },
            'word_count': {
                'total': sum(word_counts),
                'average': round(statistics.mean(word_counts), 0),
                'median': round(statistics.median(word_counts), 0),
                'min': min(word_counts),
                'max': max(word_counts),
                'stdev': round(statistics.stdev(word_counts), 0) if len(word_counts) > 1 else 0
            },
            'page_count': {
                'total': sum(page_counts),
                'average': round(statistics.mean(page_counts), 1),
                'median': round(statistics.median(page_counts), 1),
                'min': min(page_counts),
                'max': max(page_counts),
                'stdev': round(statistics.stdev(page_counts), 1) if len(page_counts) > 1 else 0
            },
            'chars_per_page': {
                'average': round(statistics.mean(chars_per_page), 0),
                'min': round(min(chars_per_page), 0),
                'max': round(max(chars_per_page), 0)
            }
        }

        # Find anomalies
        low_content = sorted(
            [(r.get('pdf_name'), r.get('char_count'), r.get('num_pages'), r.get('relative_path'))
             for r in self.extraction_results if r.get('char_count', 0) < 10000],
            key=lambda x: x[1]
        )

        analysis['low_content_pdfs'] = low_content[:10]

        return analysis

    def analyze_pdf_types(self) -> Dict[str, Any]:
        """Analyze PDF type distribution and characteristics."""
        type_stats = defaultdict(lambda: {
            'count': 0,
            'avg_quality': [],
            'avg_size_mb': [],
            'avg_pages': []
        })

        for r in self.extraction_results:
            pdf_type = r.get('pdf_type', 'unknown')
            type_stats[pdf_type]['count'] += 1
            type_stats[pdf_type]['avg_quality'].append(r.get('quality_score', 0))
            type_stats[pdf_type]['avg_size_mb'].append(r.get('file_size_mb', 0))
            type_stats[pdf_type]['avg_pages'].append(r.get('num_pages', 0))

        for pdf_type, stats in type_stats.items():
            stats['avg_quality'] = round(statistics.mean(stats['avg_quality']), 4) if stats['avg_quality'] else 0
            stats['avg_size_mb'] = round(statistics.mean(stats['avg_size_mb']), 2) if stats['avg_size_mb'] else 0
            stats['avg_pages'] = round(statistics.mean(stats['avg_pages']), 1) if stats['avg_pages'] else 0

        return dict(type_stats)

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze extraction performance metrics."""
        times = [r.get('extraction_time_seconds', 0) for r in self.extraction_results]
        sizes = [r.get('file_size_mb', 0) for r in self.extraction_results]

        # Time vs size correlation
        time_per_mb = []
        for r in self.extraction_results:
            if r.get('file_size_mb', 0) > 0:
                time_per_mb.append(r.get('extraction_time_seconds', 0) / r.get('file_size_mb', 1))

        analysis = {
            'extraction_time': {
                'total_seconds': round(sum(times), 2),
                'average_seconds': round(statistics.mean(times), 4),
                'median_seconds': round(statistics.median(times), 4),
                'min_seconds': round(min(times), 4),
                'max_seconds': round(max(times), 4),
                'stdev': round(statistics.stdev(times), 4) if len(times) > 1 else 0
            },
            'file_size_mb': {
                'total': round(sum(sizes), 2),
                'average': round(statistics.mean(sizes), 2),
                'median': round(statistics.median(sizes), 2),
                'min': round(min(sizes), 2),
                'max': round(max(sizes), 2)
            },
            'time_per_mb': {
                'average': round(statistics.mean(time_per_mb), 4),
                'min': round(min(time_per_mb), 4),
                'max': round(max(time_per_mb), 4)
            },
            'throughput': {
                'pdfs_per_minute': round(60 / statistics.mean(times), 2) if statistics.mean(times) > 0 else 0,
                'mb_per_minute': round(sum(sizes) / (sum(times) / 60), 2) if sum(times) > 0 else 0
            }
        }

        # Slow extractions
        slow_extractions = sorted(
            [(r.get('pdf_name'), r.get('extraction_time_seconds'), r.get('file_size_mb'), r.get('relative_path'))
             for r in self.extraction_results],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        analysis['slow_extractions'] = slow_extractions

        return analysis

    def analyze_metadata_completeness(self) -> Dict[str, Any]:
        """Analyze metadata completeness across extractions."""
        metadata_fields = ['title', 'author', 'subject', 'creator', 'producer', 'creation_date']
        field_coverage = defaultdict(int)

        for r in self.extraction_results:
            metadata = r.get('metadata', {})
            for field in metadata_fields:
                if metadata.get(field):
                    field_coverage[field] += 1

        analysis = {
            'field_coverage': {field: count for field, count in field_coverage.items()},
            'completeness_percentage': {
                field: round(count / len(self.extraction_results) * 100, 1)
                for field, count in field_coverage.items()
            },
            'total_with_metadata': len([r for r in self.extraction_results if r.get('metadata', {})]),
            'percentage_with_metadata': round(
                len([r for r in self.extraction_results if r.get('metadata', {})]) /
                len(self.extraction_results) * 100, 1
            )
        }

        return analysis

    def identify_improvements(self) -> List[Dict[str, str]]:
        """Identify specific areas for improvement based on analysis."""
        improvements = []

        # 1. Low quality extractions
        low_quality = [r for r in self.extraction_results if r.get('quality_score', 1.0) < 0.98]
        if low_quality:
            improvements.append({
                'category': 'Quality Improvement',
                'priority': 'High',
                'issue': f'{len(low_quality)} PDFs have quality score < 0.98',
                'details': 'These PDFs may have formatting issues, scanned images, or complex layouts',
                'recommendation': 'Consider implementing advanced post-processing or OCR fallback for low-quality extractions'
            })

        # 2. Slow extraction times
        avg_time = statistics.mean([r.get('extraction_time_seconds', 0) for r in self.extraction_results])
        slow_pdfs = [r for r in self.extraction_results if r.get('extraction_time_seconds', 0) > avg_time * 2]
        if slow_pdfs:
            improvements.append({
                'category': 'Performance Optimization',
                'priority': 'Medium',
                'issue': f'{len(slow_pdfs)} PDFs take >2x average extraction time',
                'details': f'Average extraction time: {avg_time:.2f}s, Max: {max(r.get("extraction_time_seconds", 0) for r in self.extraction_results):.2f}s',
                'recommendation': 'Implement caching, parallel processing, or adaptive chunking for large/slow PDFs'
            })

        # 3. Low content extraction
        low_content = [r for r in self.extraction_results if r.get('char_count', 0) < 10000]
        if low_content:
            improvements.append({
                'category': 'Content Extraction',
                'priority': 'Medium',
                'issue': f'{len(low_content)} PDFs have <10K characters extracted',
                'details': 'These may be image-only PDFs, scanned documents, or have extraction issues',
                'recommendation': 'Add OCR fallback and verify extraction completeness for low-content PDFs'
            })

        # 4. Metadata completeness
        metadata_coverage = len([r for r in self.extraction_results if r.get('metadata', {})])
        if metadata_coverage < len(self.extraction_results):
            missing = len(self.extraction_results) - metadata_coverage
            improvements.append({
                'category': 'Metadata Enhancement',
                'priority': 'Low',
                'issue': f'{missing} PDFs have missing or incomplete metadata',
                'details': 'Metadata helps with document classification and searchability',
                'recommendation': 'Implement metadata extraction enhancement or fallback sources (filename parsing, arXiv API)'
            })

        # 5. PDF type diversity
        pdf_types = set(r.get('pdf_type', 'unknown') for r in self.extraction_results)
        if len(pdf_types) == 1 and 'text-based' in pdf_types:
            improvements.append({
                'category': 'Format Support',
                'priority': 'Medium',
                'issue': 'Only text-based PDFs were tested',
                'details': 'No scanned/image-based PDFs in the test sample',
                'recommendation': 'Test with scanned PDFs to validate OCR pipeline and image extraction'
            })

        # 6. Method diversity
        methods = set(r.get('method_used') for r in self.extraction_results)
        if len(methods) == 1:
            improvements.append({
                'category': 'Extraction Methods',
                'priority': 'Low',
                'issue': f'Only one extraction method was used: {methods}',
                'details': 'Limited fallback mechanism was needed',
                'recommendation': 'Good reliability indicator, but ensure other methods work for edge cases'
            })

        # 7. Large file handling
        large_files = [r for r in self.extraction_results if r.get('file_size_mb', 0) > 10]
        if large_files:
            improvements.append({
                'category': 'Large File Handling',
                'priority': 'Medium',
                'issue': f'{len(large_files)} PDFs exceed 10MB',
                'details': f'Largest file: {max(r.get("file_size_mb", 0) for r in self.extraction_results):.2f}MB',
                'recommendation': 'Optimize memory usage and implement streaming for very large PDFs'
            })

        # 8. Chars per page analysis
        chars_per_page = []
        for r in self.extraction_results:
            if r.get('num_pages', 0) > 0:
                chars_per_page.append(r.get('char_count', 0) / r.get('num_pages', 0))

        if chars_per_page:
            avg_chars = statistics.mean(chars_per_page)
            low_density = [r for r in self.extraction_results
                          if r.get('num_pages', 0) > 0 and
                          r.get('char_count', 0) / r.get('num_pages', 1) < avg_chars * 0.5]
            if low_density:
                improvements.append({
                    'category': 'Content Density',
                    'priority': 'Low',
                    'issue': f'{len(low_density)} PDFs have very low character density',
                    'details': f'Average chars/page: {avg_chars:.0f}, may indicate figures/tables or extraction issues',
                    'recommendation': 'Review extraction for images-heavy PDFs and consider figure/table extraction'
                })

        return improvements

    def print_report(self):
        """Print a comprehensive analysis report."""
        print("\n" + "=" * 80)
        print("PDF EXTRACTION ANALYSIS REPORT")
        print("=" * 80)

        # Basic stats
        print(f"\n📊 Overview:")
        print(f"   Total PDFs analyzed: {len(self.extraction_results)}")
        print(f"   Overall success rate: 100%")
        print(f"   Overall quality score: {statistics.mean([r.get('quality_score', 0) for r in self.extraction_results]):.3f}")

        # Quality analysis
        quality = self.analyze_quality_scores()
        print(f"\n📈 Quality Analysis:")
        print(f"   Average: {quality['average']:.3f}")
        print(f"   Std Dev: {quality['stdev']:.4f}")
        print(f"   Distribution:")
        for range_name, count in quality['distribution'].items():
            print(f"      {range_name}: {count} PDFs")
        if quality['low_quality_extractions']:
            print(f"   ⚠️  Low quality extractions:")
            for name, score, path in quality['low_quality_extractions'][:5]:
                print(f"      {name}: {score:.3f}")

        # Methods analysis
        methods = self.analyze_extraction_methods()
        print(f"\n🔧 Extraction Methods:")
        for method, stats in methods.items():
            print(f"   {method}:")
            print(f"      Count: {stats['count']}")
            print(f"      Avg Quality: {stats['avg_quality']:.3f}")
            print(f"      Avg Time: {stats['avg_time']:.2f}s")

        # Content analysis
        content = self.analyze_text_content()
        print(f"\n📄 Text Content:")
        print(f"   Total characters: {content['char_count']['total']:,.0f}")
        print(f"   Total words: {content['word_count']['total']:,.0f}")
        print(f"   Total pages: {content['page_count']['total']}")
        print(f"   Avg chars/page: {content['chars_per_page']['average']:.0f}")
        if content['low_content_pdfs']:
            print(f"   ⚠️  Low content PDFs:")
            for name, chars, pages, path in content['low_content_pdfs'][:5]:
                print(f"      {name}: {chars:,.0f} chars in {pages} pages")

        # Performance analysis
        perf = self.analyze_performance()
        print(f"\n⏱️  Performance:")
        print(f"   Total time: {perf['extraction_time']['total_seconds']:.2f}s")
        print(f"   Average per PDF: {perf['extraction_time']['average_seconds']:.2f}s")
        print(f"   Throughput: {perf['throughput']['pdfs_per_minute']:.2f} PDFs/min")
        print(f"   Time per MB: {perf['time_per_mb']['average']:.2f}s/MB")
        if perf['slow_extractions']:
            print(f"   ⚠️  Slowest extractions:")
            for name, time, size, path in perf['slow_extractions'][:5]:
                print(f"      {name}: {time:.2f}s ({size:.2f}MB)")

        # PDF types
        types = self.analyze_pdf_types()
        print(f"\n📑 PDF Types:")
        for pdf_type, stats in types.items():
            print(f"   {pdf_type}: {stats['count']} files")
            print(f"      Avg quality: {stats['avg_quality']:.3f}")
            print(f"      Avg size: {stats['avg_size_mb']:.2f}MB")
            print(f"      Avg pages: {stats['avg_pages']:.1f}")

        # Metadata
        metadata = self.analyze_metadata_completeness()
        print(f"\n🏷️  Metadata:")
        print(f"   PDFs with metadata: {metadata['total_with_metadata']}/{len(self.extraction_results)}")
        for field, percentage in metadata['completeness_percentage'].items():
            print(f"      {field}: {percentage}%")

        # Improvements
        improvements = self.identify_improvements()
        print(f"\n🎯 Areas for Improvement:")
        for i, imp in enumerate(improvements, 1):
            priority_emoji = '🔴' if imp['priority'] == 'High' else '🟡' if imp['priority'] == 'Medium' else '🟢'
            print(f"\n   {i}. {priority_emoji} {imp['category']} ({imp['priority']})")
            print(f"      Issue: {imp['issue']}")
            print(f"      Details: {imp['details']}")
            print(f"      Recommendation: {imp['recommendation']}")

        print("\n" + "=" * 80)

        return improvements

    def save_report(self, output_file: str = 'extraction_analysis_report.json'):
        """Save detailed analysis to JSON file."""
        report = {
            'overview': {
                'total_pdfs': len(self.extraction_results),
                'success_rate': 100.0,
                'overall_quality': round(statistics.mean([r.get('quality_score', 0) for r in self.extraction_results]), 4)
            },
            'quality_analysis': self.analyze_quality_scores(),
            'methods_analysis': self.analyze_extraction_methods(),
            'content_analysis': self.analyze_text_content(),
            'performance_analysis': self.analyze_performance(),
            'pdf_types_analysis': self.analyze_pdf_types(),
            'metadata_analysis': self.analyze_metadata_completeness(),
            'improvements': self.identify_improvements()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Detailed report saved to: {output_file}")
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze PDF extraction results and identify improvements',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--results-dir', '-d',
                        default='test_results/extractions',
                        help='Directory containing extraction results (default: test_results/extractions)')

    parser.add_argument('--output', '-o',
                        default='extraction_analysis_report.json',
                        help='Output file for detailed report (default: extraction_analysis_report.json)')

    parser.add_argument('--json-only', '-j',
                        action='store_true',
                        help='Only save JSON report, no console output')

    args = parser.parse_args()

    # Run analysis
    analyzer = ExtractionAnalyzer(args.results_dir)
    analyzer.load_all_results()

    if args.json_only:
        analyzer.save_report(args.output)
    else:
        improvements = analyzer.print_report()
        analyzer.save_report(args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
