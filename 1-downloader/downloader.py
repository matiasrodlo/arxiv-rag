#!/usr/bin/env python3
"""
Fast parallel download of arXiv CS PDFs optimized for 8TB SSD
Organizes files by category/year_month for filesystem efficiency and RAG preparation
"""

import os
import subprocess
import sys
import time
import json
import signal
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
import threading
from collections import defaultdict
from datetime import datetime, timedelta

# Target categories for download
TARGET_CATEGORIES = {
    'cs.IR': 'Information Retrieval',
    'cs.CL': 'Computation & Language',
    'cs.MA': 'Multi-Agent Systems',
    'cs.SE': 'Software Engineering',
    'cs.AI': 'Artificial Intelligence',
    'cs.LG': 'Machine Learning',
    'cs.CV': 'Computer Vision',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.MM': 'Multimedia',
    'cs.CR': 'Cryptography and Security',
}

# Debug logging helper
def debug_log(location, message, data=None, hypothesis_id=None):
    """Write debug log to NDJSON file"""
    log_path = Path(__file__).parent / '.cursor' / 'debug.log'
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            log_entry = {
                'timestamp': int(time.time() * 1000),
                'location': location,
                'message': message,
                'data': data or {},
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': hypothesis_id
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass  # Silently fail if logging doesn't work

def get_paper_path(base_dir, category, paper_id):
    """
    Generate hierarchical path: {category}/{year_month}/{paper_id}.pdf
    This structure:
    - Avoids filesystem bottlenecks (max ~1000-5000 files per directory)
    - Enables efficient category-based RAG indexing
    - Preserves temporal organization
    """
    if '.' not in paper_id:
        return None
    
    parts = paper_id.split('.')
    if len(parts) != 2:
        return None
    
    year_month = parts[0]
    category_dir = Path(base_dir) / category / year_month
    return category_dir / f"{paper_id}.pdf"

def load_existing_files(base_dir, categories):
    """
    Efficiently scan existing files in hierarchical structure
    Returns: set of (category, paper_id) tuples
    """
    existing = set()
    count = 0
    base_path = Path(base_dir)
    
    print(f"Scanning existing PDFs in {base_dir}...")
    sys.stdout.flush()
    
    try:
        for category in categories:
            category_path = base_path / category
            if not category_path.exists():
                continue
            
            try:
                # Scan all year_month subdirectories
                for year_month_dir in category_path.iterdir():
                    if not year_month_dir.is_dir():
                        continue
                    
                    try:
                        for pdf_file in year_month_dir.glob("*.pdf"):
                            try:
                                # Verify file is valid and not empty
                                if pdf_file.exists() and pdf_file.stat().st_size > 0:
                                    paper_id = pdf_file.stem
                                    existing.add((category, paper_id))
                                    count += 1
                                    if count % 10000 == 0:
                                        print(f"  Scanned {count:,} existing PDFs...", end='\r')
                                        sys.stdout.flush()
                            except (OSError, PermissionError) as e:
                                # Skip files we can't read
                                continue
                    except (OSError, PermissionError) as e:
                        # Skip directories we can't read
                        continue
            except (OSError, PermissionError) as e:
                # Skip categories we can't read
                print(f"Warning: Could not scan category {category}: {e}")
                continue
    except Exception as e:
        print(f"Warning: Error during file scanning: {e}, continuing with partial results")
    
    print(f"\nFound {len(existing):,} existing PDFs across {len(categories)} categories")
    sys.stdout.flush()
    return existing

# Thread-safe metadata writing lock (per category)
_metadata_locks = defaultdict(Lock)

def save_metadata(base_dir, category, metadata):
    """Save metadata JSON for RAG indexing (thread-safe)"""
    metadata_dir = Path(base_dir) / '_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"{category}_papers.jsonl"
    
    # Use per-category lock to prevent corruption
    lock = _metadata_locks[category]
    try:
        with lock:
            with open(metadata_file, 'a') as f:
                f.write(json.dumps(metadata) + '\n')
    except Exception as e:
        # Non-critical, just log
        debug_log('downloader.py:save_metadata', 'Failed to save metadata', {'error': str(e)})

def check_gsutil_available():
    """Check if gsutil is installed and available"""
    if not shutil.which('gsutil'):
        print("ERROR: gsutil is not installed or not in PATH")
        print("Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)
    
    # Verify gsutil works
    try:
        result = subprocess.run(['gsutil', '--version'], 
                              capture_output=True, 
                              timeout=30,
                              text=True)
        if result.returncode != 0:
            print("ERROR: gsutil is installed but not working properly")
            print(f"Error: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: gsutil version check timed out")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not verify gsutil: {e}")
        sys.exit(1)

def check_disk_space(output_dir, estimated_files, avg_size_mb=2.5, warn_only=False):
    """Check if there's enough disk space"""
    try:
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)
        estimated_gb = (estimated_files * avg_size_mb) / 1024
        
        if not warn_only:
            print(f"Disk space check:")
            print(f"  Available: {free_gb:.2f} GB")
            print(f"  Estimated needed: {estimated_gb:.2f} GB")
        
        if free_gb < estimated_gb * 1.2:  # 20% buffer
            if free_gb < estimated_gb * 0.05:  # Less than 5% of needed
                return False, "CRITICAL: Less than 5% of needed space available"
            elif free_gb < estimated_gb * 0.10:  # Less than 10% of needed
                return True, f"WARNING: Only {free_gb:.2f} GB available (< 10% of needed)"
            else:
                return True, f"WARNING: Low disk space ({free_gb:.2f} GB available)"
        
        return True, None
    except Exception as e:
        return True, f"WARNING: Could not check disk space: {e}"

def setup_logging(log_dir):
    """Setup file-based logging for multi-day downloads"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Main log file
    log_file = log_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_file

def download_batch_parallel(paper_ids_file, output_dir, categories=None, max_workers=12, start_from=0, resume_state_file=None, 
                            failed_file=None, log_dir=None, check_disk_every=1000):
    """
    Download PDFs with hierarchical organization optimized for SSD and RAG
    
    Args:
        paper_ids_file: File with paper IDs (format: category|paper_id or just paper_id)
        output_dir: Base output directory
        categories: List of categories to download (None = all in TARGET_CATEGORIES)
        max_workers: Number of parallel downloads (increased for SSD)
        start_from: Start position in list
        resume_state_file: Path to state file for resume capability
        failed_file: Path to save failed downloads for retry
        log_dir: Directory for log files (None = no file logging)
        check_disk_every: Check disk space every N files (0 = disable periodic checks)
    """
    
    # Setup logging
    logger = None
    if log_dir:
        logger, log_file = setup_logging(log_dir)
        logger.info(f"Starting download session - Log file: {log_file}")
    else:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger(__name__)
    
    # Setup tracking files for successful and failed downloads
    failed_downloads = []
    successful_downloads = []
    failed_lock = Lock()
    successful_lock = Lock()
    
    if failed_file is None:
        failed_file = str(Path(output_dir) / '_failed_downloads.jsonl')
    
    # Setup successful downloads file
    successful_file = str(Path(output_dir) / '_successful_downloads.jsonl')
    
    # Initialize tracking files (append mode, so we can resume)
    failed_path = Path(failed_file)
    successful_path = Path(successful_file)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    successful_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Graceful shutdown handling
    shutdown_event = Event()
    shutdown_lock = Lock()
    shutdown_initiated = False
    
    # Check prerequisites
    check_gsutil_available()
    
    if categories is None:
        categories = list(TARGET_CATEGORIES.keys())
    
    # Validate categories
    invalid_categories = [c for c in categories if c not in TARGET_CATEGORIES]
    if invalid_categories:
        print(f"ERROR: Invalid categories: {invalid_categories}")
        print(f"Valid categories: {list(TARGET_CATEGORIES.keys())}")
        sys.exit(1)
    
    if not os.path.exists(paper_ids_file):
        print(f"ERROR: Paper IDs file not found: {paper_ids_file}")
        sys.exit(1)
    
    # Load paper IDs with category information
    print(f"Loading paper IDs from {paper_ids_file}...")
    sys.stdout.flush()
    
    papers_by_category = defaultdict(list)
    total_lines = 0
    error_lines = 0
    
    try:
        with open(paper_ids_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_lines += 1
                    
                    # Support two formats:
                    # 1. "category|paper_id" (explicit category)
                    # 2. "paper_id" (will be assigned to all categories if matches)
                    if '|' in line:
                        parts = line.split('|', 1)
                        if len(parts) == 2:
                            category, paper_id = parts
                            if category in categories:
                                papers_by_category[category].append(paper_id)
                        else:
                            error_lines += 1
                            if error_lines <= 5:
                                print(f"Warning: Invalid line format at line {line_num}: {line[:50]}")
                    else:
                        # No category specified - add to all matching categories
                        # This assumes the file contains papers from target categories
                        paper_id = line
                        for category in categories:
                            papers_by_category[category].append(paper_id)
                except Exception as e:
                    error_lines += 1
                    if error_lines <= 5:
                        print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        if error_lines > 0:
            print(f"Warning: Skipped {error_lines} invalid lines out of {total_lines} total")
            if logger:
                logger.warning(f"Skipped {error_lines} invalid lines out of {total_lines} total")
    except IOError as e:
        print(f"ERROR: Failed to read paper IDs file: {e}")
        if logger:
            logger.error(f"Failed to read paper IDs file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error loading paper IDs: {e}")
        if logger:
            logger.error(f"Unexpected error loading paper IDs: {e}")
        sys.exit(1)
    
    # Flatten to list of (category, paper_id) tuples
    all_papers = []
    for category, paper_ids in papers_by_category.items():
        for paper_id in paper_ids:
            all_papers.append((category, paper_id))
    
    print(f"Found {len(all_papers):,} papers across {len(papers_by_category)} categories")
    for category, paper_ids in sorted(papers_by_category.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {category}: {len(paper_ids):,} papers")
    sys.stdout.flush()
    
    # Create base directory structure
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing files from tracking file (much faster than filesystem scan)
    existing = set()
    successful_tracking_file = Path(output_dir) / '_successful_downloads.jsonl'
    if successful_tracking_file.exists():
        print(f"Loading existing downloads from tracking file: {successful_tracking_file}...")
        sys.stdout.flush()
        try:
            with open(successful_tracking_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        category = data.get('category')
                        paper_id = data.get('paper_id')
                        if category and paper_id:
                            existing.add((category, paper_id))
                        if line_num % 10000 == 0:
                            print(f"  Loaded {line_num:,} records...", end='\r')
                            sys.stdout.flush()
                    except json.JSONDecodeError:
                        continue
            print(f"\nLoaded {len(existing):,} existing downloads from tracking file")
            sys.stdout.flush()
        except Exception as e:
            print(f"Warning: Could not load tracking file: {e}")
            print("Falling back to filesystem scan...")
            sys.stdout.flush()
            existing = load_existing_files(output_dir, categories)
    else:
        print("No tracking file found, scanning filesystem...")
        sys.stdout.flush()
        existing = load_existing_files(output_dir, categories)
    
    # Filter out already downloaded
    print("Filtering out already downloaded papers...")
    sys.stdout.flush()
    to_download = [(cat, pid) for cat, pid in all_papers if (cat, pid) not in existing]
    print(f"Remaining to download: {len(to_download):,}")
    sys.stdout.flush()
    
    if not to_download:
        print("All PDFs already downloaded!")
        return
    
    # Check disk space before starting
    ok, msg = check_disk_space(output_dir, len(to_download))
    if not ok:
        logger.error(msg)
        sys.exit(1)
    elif msg:
        logger.warning(msg)
        if not msg.startswith("WARNING"):
            response = input("Continue anyway? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("Aborted by user")
                sys.exit(1)
    
    # Resume capability - track offset for absolute indexing
    resume_offset = 0  # Track how many files we've skipped for absolute position
    if resume_state_file and os.path.exists(resume_state_file):
        print(f"Loading resume state from {resume_state_file}...")
        try:
            with open(resume_state_file, 'r') as f:
                resume_data = json.load(f)
                resume_index = resume_data.get('last_index', 0)
                # Resume index is absolute position in to_download list
                if resume_index > 0 and resume_index < len(to_download):
                    print(f"Resuming from index {resume_index:,} (absolute position)")
                    resume_offset = resume_index
                    to_download = to_download[resume_index:]
                elif resume_index >= len(to_download):
                    print(f"Warning: Resume index {resume_index:,} >= total files {len(to_download):,}. Starting from beginning.")
        except Exception as e:
            print(f"Warning: Could not load resume state: {e}")
    
    # Start from a specific position if specified
    if start_from > 0:
        if start_from >= len(to_download):
            print(f"Warning: start_from ({start_from:,}) is >= total files ({len(to_download):,}). Starting from beginning.")
            start_from = 0
        else:
            print(f"Starting from position {start_from:,} (skipping first {start_from:,} files)")
            resume_offset = start_from
            to_download = to_download[start_from:]
            sys.stdout.flush()
    
    # Process files with parallelism
    total_files = len(to_download)
    downloaded = 0
    failed = 0
    processed = 0
    
    # Thread-safe counters and state
    lock = Lock()
    last_saved_index = 0
    last_disk_check = 0
    start_time = time.time()
    last_progress_time = time.time()
    
    # Statistics per category (thread-safe updates required)
    category_stats = defaultdict(lambda: {'downloaded': 0, 'failed': 0})
    category_stats_lock = Lock()  # Separate lock for category stats
    
    # Define signal handler after variables are initialized
    def signal_handler(signum, frame):
        nonlocal shutdown_initiated, processed, downloaded, failed
        with shutdown_lock:
            if shutdown_initiated:
                logger.warning(f"Force shutdown requested (signal {signum})")
                sys.exit(1)
            shutdown_initiated = True
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()
        # Save final state before shutdown
        if resume_state_file:
            try:
                with lock:
                    state = {
                        'last_index': processed,
                        'downloaded': downloaded,
                        'failed': failed,
                        'processed': processed,
                        'timestamp': time.time(),
                        'datetime': datetime.now().isoformat(),
                        'shutdown': True
                    }
                temp_file = resume_state_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=2)
                os.replace(temp_file, resume_state_file)
                logger.info("Resume state saved before shutdown")
            except Exception as e:
                logger.warning(f"Failed to save state before shutdown: {e}")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting parallel download of {total_files:,} files ({max_workers} concurrent workers)...")
    logger.info(f"Output structure: {{category}}/{{year_month}}/{{paper_id}}.pdf")
    logger.info(f"Successful downloads will be saved to: {successful_file}")
    logger.info(f"Failed downloads will be saved to: {failed_file}")
    if check_disk_every > 0:
        logger.info(f"Disk space will be checked every {check_disk_every:,} files")
    
    def save_successful_download(category, paper_id, version, pdf_path, year_month):
        """Thread-safe save of successful download"""
        entry = {
            'category': category,
            'paper_id': paper_id,
            'version': version,
            'year_month': year_month,
            'path': str(pdf_path.relative_to(base_path)) if pdf_path else None,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        with successful_lock:
            try:
                with open(successful_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
            except Exception as e:
                logger.warning(f"Failed to save successful download record: {e}")
    
    def save_failed_download(category, paper_id, error_msg=None, attempts=4):
        """Thread-safe save of failed download"""
        entry = {
            'category': category,
            'paper_id': paper_id,
            'error': error_msg,
            'attempts': attempts,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        with failed_lock:
            try:
                with open(failed_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
            except Exception as e:
                logger.warning(f"Failed to save failed download record: {e}")
    
    def download_single_file(category, paper_id, idx, global_idx):
        """Download a single file with version retry logic"""
        nonlocal downloaded, failed, processed, last_saved_index
        
        debug_log('downloader.py:download_single_file', 'entry', {
            'category': category,
            'paper_id': paper_id,
            'idx': idx,
            'global_idx': global_idx,
            'thread_id': threading.current_thread().ident
        }, 'E')
        
        if '.' not in paper_id:
            with lock:
                failed += 1
                processed += 1
            with category_stats_lock:
                category_stats[category]['failed'] += 1
            return False
        
        parts = paper_id.split('.')
        if len(parts) != 2:
            with lock:
                failed += 1
                processed += 1
            with category_stats_lock:
                category_stats[category]['failed'] += 1
            return False
        
        year_month = parts[0]
        pdf_path = get_paper_path(output_dir, category, paper_id)
        
        if pdf_path is None:
            with lock:
                failed += 1
                processed += 1
            with category_stats_lock:
                category_stats[category]['failed'] += 1
            return False
        
        # Ensure directory exists
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists (check with lock to prevent race condition)
        # Use try-except for atomic check-and-skip
        try:
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                with lock:
                    downloaded += 1
                    processed += 1
                with category_stats_lock:
                    category_stats[category]['downloaded'] += 1
                if global_idx % 1000 == 0:
                    print(f"  [{global_idx:,}] Skipped (exists): {category}/{paper_id}")
                    sys.stdout.flush()
                return True
        except (OSError, FileNotFoundError):
            # File disappeared between check and stat, continue to download
            pass
        
        # Try versions in order: v1, v2, v3, v4
        success = False
        downloaded_version = None
        max_retries = 2  # Retry each version up to 2 times
        download_timeout = 120  # Increased timeout for large files
        
        for version in ['v1', 'v2', 'v3', 'v4']:
            if success:
                break
                
            gs_path = f"gs://arxiv-dataset/arxiv/pdf/{year_month}/{paper_id}{version}.pdf"
            
            if global_idx <= 10 or global_idx % 1000 == 0:
                debug_log('downloader.py:download', f'Downloading file {global_idx}', {
                    'category': category,
                    'paper_id': paper_id,
                    'gs_path': gs_path,
                    'version': version,
                    'output_path': str(pdf_path)
                }, 'A')
            
            cmd = ['gsutil', 'cp', gs_path, str(pdf_path)]
            
            # Retry logic for network issues
            for retry in range(max_retries + 1):
                if success:
                    break
                    
                try:
                    # Use Popen with process group for proper cleanup on timeout
                    process = subprocess.Popen(
                        cmd,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                    )
                    
                    try:
                        stdout, stderr = process.communicate(timeout=download_timeout)
                        returncode = process.returncode
                    except subprocess.TimeoutExpired:
                        # Kill the entire process group
                        if hasattr(os, 'setsid'):
                            try:
                                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            except (ProcessLookupError, OSError):
                                pass
                        else:
                            process.terminate()
                        
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if hasattr(os, 'setsid'):
                                try:
                                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                                except (ProcessLookupError, OSError):
                                    pass
                            else:
                                process.kill()
                            process.wait()
                        
                        # Retry on timeout unless this is the last retry
                        if retry < max_retries:
                            time.sleep(1 * (retry + 1))  # Exponential backoff
                            continue
                        else:
                            raise subprocess.TimeoutExpired(cmd, download_timeout)
                
                    if returncode == 0:
                        # Verify file exists and is not empty (with error handling)
                        try:
                            # Wait a moment for file system to sync
                            time.sleep(0.1)
                            
                            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                                # Basic PDF validation: check for PDF header
                                try:
                                    with open(pdf_path, 'rb') as f:
                                        header = f.read(4)
                                        if header == b'%PDF':
                                            # Verify file is readable and complete
                                            f.seek(-1, 2)  # Seek to end
                                            f.read(1)  # Try to read last byte
                                            
                                            with lock:
                                                downloaded += 1
                                                processed += 1
                                            with category_stats_lock:
                                                category_stats[category]['downloaded'] += 1
                                            
                                            downloaded_version = version
                                            success = True
                                            
                                            # Save successful download record
                                            save_successful_download(category, paper_id, version, pdf_path, year_month)
                                            
                                            # Save metadata for RAG
                                            metadata = {
                                                'paper_id': paper_id,
                                                'category': category,
                                                'category_name': TARGET_CATEGORIES[category],
                                                'year_month': year_month,
                                                'version': version,
                                                'path': str(pdf_path.relative_to(base_path)),
                                                'downloaded_at': time.time()
                                            }
                                            save_metadata(output_dir, category, metadata)
                                            
                                            if global_idx % 100 == 0 or global_idx <= 20:
                                                print(f"  [{global_idx:,}] ✓ {category}/{paper_id} (v{version[-1]})")
                                                sys.stdout.flush()
                                            break  # Success, exit version loop
                                        else:
                                            # Invalid PDF header - delete and try next version
                                            try:
                                                pdf_path.unlink()
                                            except:
                                                pass
                                except (IOError, OSError) as e:
                                    # File read error - delete and try next version
                                    try:
                                        if pdf_path.exists():
                                            pdf_path.unlink()
                                    except:
                                        pass
                                    if retry < max_retries:
                                        time.sleep(1 * (retry + 1))
                                        continue
                            else:
                                # Empty file or doesn't exist - delete if exists and try next version
                                if pdf_path.exists():
                                    try:
                                        pdf_path.unlink()
                                    except:
                                        pass
                                if retry < max_retries:
                                    time.sleep(1 * (retry + 1))
                                    continue
                        except (OSError, FileNotFoundError, PermissionError) as e:
                            # File operation failed, try next version
                            debug_log('downloader.py:file_check', 'File check failed', {
                                'error': str(e),
                                'paper_id': paper_id
                            })
                            if retry < max_retries:
                                time.sleep(1 * (retry + 1))
                                continue
                    elif returncode != 0:
                        # Check for specific error types that shouldn't be retried
                        error_msg = stderr.split('\n')[0] if stderr else "Unknown error"
                        if "No URLs matched" in error_msg or "No such object" in error_msg:
                            # File doesn't exist, don't retry this version
                            if global_idx <= 20 or global_idx % 1000 == 0:
                                print(f"  [{global_idx:,}] ✗ {category}/{paper_id} {version} - File not found")
                                sys.stdout.flush()
                            break  # Try next version
                        else:
                            # Network or other error, retry
                            if retry < max_retries:
                                if global_idx <= 20 or global_idx % 1000 == 0:
                                    print(f"  [{global_idx:,}] ⚠ {category}/{paper_id} {version} - Retrying ({retry+1}/{max_retries})")
                                    sys.stdout.flush()
                                time.sleep(2 * (retry + 1))  # Exponential backoff
                                continue
                            else:
                                if global_idx <= 20 or global_idx % 1000 == 0:
                                    print(f"  [{global_idx:,}] ✗ {category}/{paper_id} {version} - {error_msg[:60]}")
                                    sys.stdout.flush()
                                    
                except subprocess.TimeoutExpired:
                    debug_log('downloader.py:timeout', 'TimeoutExpired', {
                        'category': category,
                        'paper_id': paper_id,
                        'version': version,
                        'global_idx': global_idx,
                        'retry': retry
                    }, 'C')
                    if retry < max_retries:
                        time.sleep(2 * (retry + 1))
                        continue
                    else:
                        if global_idx <= 20 or global_idx % 1000 == 0:
                            print(f"  [{global_idx:,}] ⏱ Timeout: {category}/{paper_id} {version}")
                            sys.stdout.flush()
                except (ConnectionError, OSError) as e:
                    # Network errors - retry
                    if retry < max_retries:
                        if global_idx <= 20 or global_idx % 1000 == 0:
                            print(f"  [{global_idx:,}] ⚠ Network error: {category}/{paper_id} {version} - Retrying")
                            sys.stdout.flush()
                        time.sleep(3 * (retry + 1))
                        continue
                    else:
                        if global_idx <= 10 or global_idx % 1000 == 0:
                            debug_log('downloader.py:network_error', f'Network error for file {global_idx}', {
                                'category': category,
                                'paper_id': paper_id,
                                'version': version,
                                'error': str(e)
                            }, 'D')
                except Exception as e:
                    if global_idx <= 10 or global_idx % 1000 == 0:
                        debug_log('downloader.py:exception', f'Exception for file {global_idx}', {
                            'category': category,
                            'paper_id': paper_id,
                            'version': version,
                            'error': str(e),
                            'retry': retry
                        }, 'D')
                    if retry < max_retries:
                        time.sleep(1 * (retry + 1))
                        continue
                    else:
                        if global_idx <= 20 or global_idx % 1000 == 0:
                            print(f"  [{global_idx:,}] ⚠ Exception: {category}/{paper_id} {version} - {str(e)[:60]}")
                            sys.stdout.flush()
        
        if not success:
            # Save failed download record
            save_failed_download(category, paper_id, error_msg="All versions (v1-v4) failed", attempts=4)
            
            with lock:
                failed += 1
                processed += 1
            with category_stats_lock:
                category_stats[category]['failed'] += 1
            if global_idx <= 20 or global_idx % 1000 == 0:
                print(f"  [{global_idx:,}] ✗ All versions failed: {category}/{paper_id}")
                sys.stdout.flush()
        
        # Save resume state periodically (thread-safe)
        if resume_state_file and global_idx > 0:
            should_save = False
            with lock:
                # Check and update atomically to prevent race condition
                if global_idx - last_saved_index >= 500:  # Save every 500 files (more frequent)
                    should_save = True
                    last_saved_index = global_idx
                    # Capture current stats while holding lock
                    current_downloaded = downloaded
                    current_failed = failed
                    current_processed = processed
            
            if should_save:
                try:
                    state = {
                        'last_index': global_idx,
                        'downloaded': current_downloaded,
                        'failed': current_failed,
                        'processed': current_processed,
                        'timestamp': time.time(),
                        'datetime': datetime.now().isoformat()
                    }
                    # Use atomic write (write to temp file then rename)
                    temp_file = resume_state_file + '.tmp'
                    with open(temp_file, 'w') as f:
                        json.dump(state, f, indent=2)
                        # Flush and sync to ensure data is written
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_file, resume_state_file)
                except Exception as e:
                    debug_log('downloader.py:save_state', 'Failed to save resume state', {'error': str(e)})
                    logger.warning(f"Failed to save resume state: {e}")
        
        return success
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks with absolute global_idx
        future_to_paper = {
            executor.submit(download_single_file, category, paper_id, idx, resume_offset + idx): (category, paper_id, idx)
            for idx, (category, paper_id) in enumerate(to_download, 1)
        }
        
        debug_log('downloader.py:main', 'Starting as_completed loop', {
            'total_futures': len(future_to_paper),
            'max_workers': max_workers
        }, 'E')
        
        completed_count = 0
        for future in as_completed(future_to_paper):
            # Check for shutdown signal
            if shutdown_event.is_set():
                logger.warning("Shutdown signal received, waiting for current downloads to complete...")
                # Cancel remaining futures
                for f in future_to_paper:
                    if not f.done():
                        f.cancel()
                break
            
            completed_count += 1
            category, paper_id, idx = future_to_paper[future]
            
            try:
                future.result(timeout=300)  # 5 minute timeout per future
            except Exception as e:
                # Save failed download record for unexpected exceptions
                save_failed_download(category, paper_id, error_msg=f"Exception: {str(e)}", attempts=0)
                
                debug_log('downloader.py:future_exception', 'Future exception', {
                    'category': category,
                    'paper_id': paper_id,
                    'idx': idx,
                    'error': str(e),
                    'error_type': type(e).__name__
                }, 'D')
                with lock:
                    failed += 1
                    processed += 1
                with category_stats_lock:
                    category_stats[category]['failed'] += 1
            
            # Progress update every 50 files
            with lock:
                current_processed = processed
                current_downloaded = downloaded
                current_failed = failed
            
            if current_processed % 50 == 0 or current_processed == total_files:
                pct = (current_processed/total_files)*100 if total_files > 0 else 0
                elapsed = time.time() - start_time
                rate = current_processed / elapsed if elapsed > 0 else 0
                eta = (total_files - current_processed) / rate if rate > 0 else 0
                print(f"\n  📊 Progress: {current_processed:,}/{total_files:,} ({pct:.2f}%) | ✓ {current_downloaded:,} | ✗ {current_failed:,} | Rate: {rate:.1f}/s | ETA: {int(eta/60)}m")
                sys.stdout.flush()
            
            # Periodic disk space check
            if check_disk_every > 0 and current_processed > 0:
                with lock:
                    if current_processed - last_disk_check >= check_disk_every:
                        last_disk_check = current_processed
                        ok, msg = check_disk_space(output_dir, total_files - current_processed, warn_only=True)
                        if msg and not ok:
                            logger.error(f"Disk space critical: {msg}")
                            logger.warning("Consider freeing space or stopping download")
                        elif msg:
                            logger.warning(msg)
    
    # Final summary
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {downloaded:,} downloaded, {failed:,} failed")
    print(f"\nBy Category:")
    for category in sorted(categories):
        stats = category_stats[category]
        print(f"  {category:8} ({TARGET_CATEGORIES[category]:30}): "
              f"✓ {stats['downloaded']:>6,} | ✗ {stats['failed']:>6,}")
    print(f"\nOutput: {output_dir}")
    print(f"Structure: {{category}}/{{year_month}}/{{paper_id}}.pdf")
    print(f"Metadata: {output_dir}/_metadata/{{category}}_papers.jsonl")
    print(f"\n📋 Download Records:")
    print(f"  ✓ Successful: {successful_file}")
    
    # Count lines in successful file
    try:
        with open(successful_file, 'r') as f:
            successful_count = sum(1 for _ in f)
        print(f"     ({successful_count:,} records)")
    except:
        pass
    
    print(f"  ✗ Failed: {failed_file}")
    
    # Count lines in failed file
    try:
        with open(failed_file, 'r') as f:
            failed_count = sum(1 for _ in f)
        print(f"     ({failed_count:,} records)")
        if failed_count > 0:
            print(f"\n💡 To retry failed downloads:")
            print(f"   python -c \"")
            print(f"   import json")
            print(f"   with open('{failed_file}') as f:")
            print(f"       for line in f:")
            print(f"           data = json.loads(line)")
            print(f"           print(f\"{{data['category']}}|{{data['paper_id']}}\")\" > failed_retry.txt")
            print(f"   python downloader.py --ids failed_retry.txt --output {output_dir}")
    except:
        pass
    
    sys.stdout.flush()
    logger.info(f"Download complete. Successful: {downloaded:,}, Failed: {failed:,}")
    logger.info(f"Records saved to: {successful_file} and {failed_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fast parallel download of arXiv CS PDFs optimized for SSD and RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all target categories
  python downloader.py --ids papers.txt --output /Volumes/8SSD/DOIS/pdfs
  
  # Download specific categories
  python downloader.py --ids papers.txt --output /Volumes/8SSD/DOIS/pdfs --categories cs.AI cs.LG
  
  # Resume interrupted download
  python downloader.py --ids papers.txt --output /Volumes/8SSD/DOIS/pdfs --resume-state state.json
  
  # High parallelism for fast SSD
  python downloader.py --ids papers.txt --output /Volumes/8SSD/DOIS/pdfs --workers 20
        """
    )
    parser.add_argument('--ids', required=True, help='Paper IDs file (format: category|paper_id or just paper_id)')
    parser.add_argument('--output', '-o', required=True, help='Output base directory')
    parser.add_argument('--categories', nargs='+', default=None,
                       choices=list(TARGET_CATEGORIES.keys()),
                       help='Categories to download (default: all target categories)')
    parser.add_argument('--workers', type=int, default=12,
                       help='Number of parallel workers (default: 12, increase for fast SSD)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from this position in the list (for testing)')
    parser.add_argument('--resume-state', type=str, default=None,
                       help='Path to resume state file (enables resume capability)')
    parser.add_argument('--failed-file', type=str, default=None,
                       help='Path to save failed downloads (default: {output}/_failed_downloads.jsonl)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: no file logging)')
    parser.add_argument('--check-disk-every', type=int, default=1000,
                       help='Check disk space every N files (0 = disable, default: 1000)')
    
    args = parser.parse_args()
    
    download_batch_parallel(
        args.ids,
        args.output,
        categories=args.categories,
        max_workers=args.workers,
        start_from=args.start_from,
        resume_state_file=args.resume_state,
        failed_file=args.failed_file,
        log_dir=args.log_dir,
        check_disk_every=args.check_disk_every
    )
