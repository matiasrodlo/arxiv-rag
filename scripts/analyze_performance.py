#!/usr/bin/env python3
"""
Analyze performance logs from the pipeline run.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import statistics

def load_logs(log_path: str) -> List[Dict]:
    """Load NDJSON log file."""
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs

def analyze_performance(logs: List[Dict]) -> Dict:
    """Analyze performance metrics from logs."""
    analysis = {
        'overall': {},
        'batches': [],
        'papers': [],
        'stages': defaultdict(list),
        'bottlenecks': []
    }
    
    # Extract overall metrics
    perf_start = next((log for log in logs if log.get('hypothesisId') == 'PERF_START'), None)
    perf_end = next((log for log in logs if log.get('hypothesisId') == 'PERF_END'), None)
    
    if perf_start and perf_end:
        analysis['overall'] = {
            'total_papers': perf_end['data'].get('total_papers', 0),
            'successful': perf_end['data'].get('successful', 0),
            'failed': perf_end['data'].get('failed', 0),
            'total_duration_seconds': perf_end['data'].get('total_duration_seconds', 0),
            'papers_per_second': perf_end['data'].get('papers_per_second', 0),
            'papers_per_hour': perf_end['data'].get('papers_per_hour', 0),
            'initial_memory_mb': perf_start['data'].get('initial_memory_mb', 0),
            'final_memory_mb': perf_end['data'].get('final_memory_mb', 0),
            'num_workers': perf_start['data'].get('num_workers', 0),
            'batch_size': perf_start['data'].get('batch_size', 0)
        }
    
    # Extract batch metrics
    batch_starts = [log for log in logs if log.get('hypothesisId') == 'BATCH_START']
    batch_ends = [log for log in logs if log.get('hypothesisId') == 'BATCH_END']
    
    for start, end in zip(batch_starts, batch_ends):
        analysis['batches'].append({
            'batch_idx': start['data'].get('batch_idx', 0),
            'batch_size': start['data'].get('batch_size', 0),
            'duration_seconds': end['data'].get('duration_seconds', 0),
            'papers_per_second': end['data'].get('papers_per_second', 0),
            'cpu_percent_start': start['data'].get('cpu_percent', 0),
            'cpu_percent_end': end['data'].get('cpu_percent', 0),
            'memory_mb_start': start['data'].get('memory_mb', 0),
            'memory_mb_end': end['data'].get('memory_mb', 0)
        })
    
    # Extract paper metrics
    paper_starts = [log for log in logs if log.get('hypothesisId') == 'PAPER_START']
    paper_ends = [log for log in logs if log.get('hypothesisId') == 'PAPER_END']
    
    paper_times = {}
    for start in paper_starts:
        paper_id = start['data'].get('paper_id')
        if paper_id:
            paper_times[paper_id] = {'start': start['timestamp']}
    
    for end in paper_ends:
        paper_id = end['data'].get('paper_id')
        if paper_id and paper_id in paper_times:
            paper_times[paper_id]['end'] = end['timestamp']
            paper_times[paper_id]['duration'] = end['data'].get('total_duration_seconds', 0)
            paper_times[paper_id]['num_chunks'] = end['data'].get('num_chunks', 0)
            paper_times[paper_id]['text_length'] = end['data'].get('text_length', 0)
    
    analysis['papers'] = list(paper_times.values())
    
    # Extract stage metrics
    extraction_logs = [log for log in logs if log.get('hypothesisId') == 'STAGE_EXTRACTION']
    cleaning_logs = [log for log in logs if log.get('hypothesisId') == 'STAGE_CLEANING']
    chunking_logs = [log for log in logs if log.get('hypothesisId') == 'STAGE_CHUNKING']
    
    analysis['stages']['extraction'] = [log['data'].get('duration_seconds', 0) for log in extraction_logs]
    analysis['stages']['cleaning'] = [log['data'].get('duration_seconds', 0) for log in cleaning_logs]
    analysis['stages']['chunking'] = [log['data'].get('duration_seconds', 0) for log in chunking_logs]
    
    # Identify bottlenecks
    if analysis['stages']['extraction']:
        avg_extraction = statistics.mean(analysis['stages']['extraction'])
        analysis['bottlenecks'].append(('extraction', avg_extraction))
    
    if analysis['stages']['cleaning']:
        avg_cleaning = statistics.mean(analysis['stages']['cleaning'])
        analysis['bottlenecks'].append(('cleaning', avg_cleaning))
    
    if analysis['stages']['chunking']:
        avg_chunking = statistics.mean(analysis['stages']['chunking'])
        analysis['bottlenecks'].append(('chunking', avg_chunking))
    
    analysis['bottlenecks'].sort(key=lambda x: x[1], reverse=True)
    
    return analysis

def print_report(analysis: Dict):
    """Print performance report."""
    print("=" * 80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Overall metrics
    overall = analysis['overall']
    if overall:
        print("OVERALL PERFORMANCE")
        print("-" * 80)
        print(f"Total Papers:        {overall.get('total_papers', 0)}")
        print(f"Successful:          {overall.get('successful', 0)}")
        print(f"Failed:               {overall.get('failed', 0)}")
        print(f"Total Duration:      {overall.get('total_duration_seconds', 0):.2f} seconds ({overall.get('total_duration_seconds', 0)/60:.2f} minutes)")
        print(f"Throughput:          {overall.get('papers_per_second', 0):.2f} papers/second")
        print(f"                     {overall.get('papers_per_hour', 0):.2f} papers/hour")
        print(f"Workers:             {overall.get('num_workers', 0)}")
        print(f"Batch Size:          {overall.get('batch_size', 0)}")
        print(f"Memory Usage:        {overall.get('initial_memory_mb', 0):.0f} MB → {overall.get('final_memory_mb', 0):.0f} MB")
        print()
    
    # Batch metrics
    if analysis['batches']:
        print("BATCH PERFORMANCE")
        print("-" * 80)
        batch_durations = [b['duration_seconds'] for b in analysis['batches']]
        batch_throughput = [b['papers_per_second'] for b in analysis['batches']]
        avg_cpu = statistics.mean([(b['cpu_percent_start'] + b['cpu_percent_end']) / 2 for b in analysis['batches'] if b['cpu_percent_start'] > 0])
        
        print(f"Number of Batches:   {len(analysis['batches'])}")
        print(f"Avg Batch Duration:  {statistics.mean(batch_durations):.2f} seconds")
        print(f"Min Batch Duration:   {min(batch_durations):.2f} seconds")
        print(f"Max Batch Duration:   {max(batch_durations):.2f} seconds")
        print(f"Avg Throughput:      {statistics.mean(batch_throughput):.2f} papers/second")
        if avg_cpu > 0:
            print(f"Avg CPU Usage:       {avg_cpu:.1f}%")
        print()
    
    # Stage metrics
    print("STAGE PERFORMANCE")
    print("-" * 80)
    for stage_name, durations in analysis['stages'].items():
        if durations:
            print(f"{stage_name.capitalize():15}  Avg: {statistics.mean(durations):.3f}s  "
                  f"Min: {min(durations):.3f}s  Max: {max(durations):.3f}s  "
                  f"Total: {sum(durations):.2f}s")
    print()
    
    # Bottlenecks
    if analysis['bottlenecks']:
        print("BOTTLENECKS (by average time)")
        print("-" * 80)
        for stage, avg_time in analysis['bottlenecks']:
            print(f"{stage.capitalize():15}  {avg_time:.3f} seconds")
        print()
    
    # Paper metrics
    if analysis['papers']:
        paper_durations = [p['duration'] for p in analysis['papers'] if 'duration' in p]
        if paper_durations:
            print("PER-PAPER PERFORMANCE")
            print("-" * 80)
            print(f"Avg Paper Duration:  {statistics.mean(paper_durations):.2f} seconds")
            print(f"Min Paper Duration:  {min(paper_durations):.2f} seconds")
            print(f"Max Paper Duration:   {max(paper_durations):.2f} seconds")
            print(f"Median Duration:     {statistics.median(paper_durations):.2f} seconds")
            print()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze performance logs")
    parser.add_argument('--log-file', type=str, default='.cursor/debug.log',
                       help='Path to log file (default: .cursor/debug.log)')
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Loading logs from: {log_path}")
    logs = load_logs(str(log_path))
    print(f"Loaded {len(logs)} log entries")
    print()
    
    analysis = analyze_performance(logs)
    print_report(analysis)

if __name__ == '__main__':
    main()

