# arXiv Paper Download System

Download arXiv Computer Science papers with automatic retry, resume capability, and deduplication.

## Quick Start

```bash
./run_download.sh
./scripts/health_monitor.sh
./scripts/status.sh
```

## Download Process

Downloads 105,421 papers across 10 CS categories, achieving 89.7% success rate (94,520 papers downloaded).

**Categories:**
- cs.AI (17,899), cs.LG (31,093), cs.CV (21,302), cs.CL (8,052), cs.CR (7,757)
- cs.SE (5,815), cs.DC (5,372), cs.IR (4,588), cs.MA (1,839), cs.MM (1,704)

**Process:**
1. Parallel download: 40 workers fetch from Google Cloud Storage
2. Organization: `{category}/{year_month}/{paper_id}.pdf`
3. Deduplication: Removes cross-listed duplicates (24,489 files, 79.75 GB freed)
4. Result: 70,016 unique papers in 256 GB

## Files

**Core:**
- `run_download.sh` - Main orchestrator
- `downloader.py` - Download engine
- `data/papers_with_dois.txt` - Paper list (105,421 papers)
- `data/download_state.json` - Resume state

**Utilities (`scripts/`):**
- `health_monitor.sh` - Real-time monitoring
- `status.sh` - Status check
- `retry_failures.sh` - Retry failed downloads
- `deduplicate_papers.py` - Remove duplicates

## Configuration

Edit `run_download.sh`:

```bash
MAX_WORKERS=40      # Parallel workers (30-50 recommended)
MAX_RETRIES=3       # Retry attempts
RETRY_DELAY=300     # Seconds between retries
```

## Common Operations

**Resume:**
```bash
./run_download.sh
```

**Retry failures:**
```bash
./scripts/retry_failures.sh
```

**Deduplicate:**
```bash
python3 scripts/deduplicate_papers.py --pdfs-dir pdfs --execute --update-tracking
```

**Stop:**
Press `Ctrl+C` (saves state and exits cleanly)

## Statistics

| Metric | Value |
|--------|-------|
| Target papers | 105,421 |
| Downloaded | 94,520 (89.7%) |
| Unique papers | 70,016 |
| Duplicates removed | 24,489 |
| Space freed | 79.75 GB |
| Final size | 256 GB |

## Prerequisites

- Python 3.7+
- Google Cloud SDK (gsutil)
- 300+ GB disk space
- Stable network connection

**Test access:**
```bash
python3 --version
gsutil version
gsutil ls gs://arxiv-dataset/arxiv/pdf/
```

## Troubleshooting

**Download not starting:**
```bash
rm -f .download.lock
```

**Check status:**
```bash
./scripts/status.sh
tail -f logs/wrapper.log
```

**High failure rate:**
Reduce workers in `run_download.sh` (set `MAX_WORKERS=20`)

## License

Downloads from arXiv dataset on Google Cloud Storage. Respect arXiv's terms: https://arxiv.org/help/api/tou
