#!/usr/bin/env python3
"""
Check memory optimization status and recommendations.
Shows how to use idle RAM to unlock CPU bottlenecks.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.core.memory_optimizer import MemoryOptimizer
    import yaml
    from loguru import logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def main():
    """Check memory optimization status."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("=" * 80)
    print("MEMORY OPTIMIZATION CHECK")
    print("=" * 80)
    print()
    
    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize memory optimizer
    mem_config = config.get('memory_optimization', {})
    optimizer = MemoryOptimizer(
        use_ram_disk=mem_config.get('use_ram_disk', True),
        ram_disk_size_gb=mem_config.get('ram_disk_size_gb', 20),
        enable_model_caching=mem_config.get('enable_model_caching', True),
        max_workers=config.get('processing', {}).get('num_workers', 12)
    )
    
    # Get memory stats
    print("📊 MEMORY STATUS")
    print("-" * 80)
    stats = optimizer.get_memory_stats()
    if stats:
        print(f"  Total RAM:        {stats.get('system_total_gb', 0):.1f} GB")
        print(f"  Available RAM:    {stats.get('system_available_gb', 0):.1f} GB")
        print(f"  Used RAM:         {100 - stats.get('system_free_percent', 0):.1f}%")
        print(f"  Free RAM:         {stats.get('system_free_percent', 0):.1f}%")
        print()
        
        available_gb = stats.get('system_available_gb', 0)
        if available_gb > 50:
            print(f"  ✅ You have {available_gb:.1f}GB of idle RAM available!")
            print(f"     This can be used to reduce CPU bottlenecks.")
        elif available_gb > 20:
            print(f"  ⚠️  You have {available_gb:.1f}GB available (moderate)")
        else:
            print(f"  ⚠️  Low RAM available: {available_gb:.1f}GB")
    else:
        print("  ⚠️  Could not get memory stats (install psutil: pip install psutil)")
    print()
    
    # Check RAM disk
    print("💾 RAM DISK STATUS")
    print("-" * 80)
    ram_disk = optimizer.setup_ram_disk()
    if ram_disk:
        print(f"  ✅ RAM disk available: {ram_disk}")
        print(f"     PDF extraction cache will use this for faster I/O")
    else:
        print(f"  ⚠️  RAM disk not available (using /tmp cache instead)")
    print()
    
    # Worker recommendations
    print("👷 WORKER RECOMMENDATIONS")
    print("-" * 80)
    current_workers = config.get('processing', {}).get('num_workers', 12)
    recommended = optimizer.recommend_worker_count()
    
    print(f"  Current workers:   {current_workers}")
    print(f"  Recommended:       {recommended}")
    
    if recommended > current_workers:
        print()
        print(f"  💡 RECOMMENDATION: Increase workers to {recommended}")
        print(f"     You have enough RAM to support more workers!")
        print(f"     This will better utilize your CPU cores.")
        print()
        print(f"     To apply: Update config.yaml:")
        print(f"       processing:")
        print(f"         num_workers: {recommended}")
    elif recommended < current_workers:
        print()
        print(f"  ⚠️  You may be using too many workers for available RAM")
        print(f"     Consider reducing to {recommended} for better stability")
    else:
        print()
        print(f"  ✅ Worker count is optimal for your system")
    print()
    
    # Optimization tips
    print("🚀 OPTIMIZATION TIPS")
    print("-" * 80)
    print("  1. RAM Disk Cache:")
    print("     - PDF extraction cache uses RAM disk (faster than disk I/O)")
    print("     - Reduces CPU wait time for file operations")
    print()
    print("  2. Larger Batch Sizes:")
    print("     - Embedding batch size increased automatically")
    print("     - More data processed in memory = fewer CPU cycles wasted")
    print()
    print("  3. Model Caching:")
    print("     - Models stay in memory (reduces reload overhead)")
    print("     - Each worker still loads its own models (Python limitation)")
    print()
    print("  4. More Workers (if RAM allows):")
    if recommended > current_workers:
        print(f"     - You can safely use {recommended} workers")
        print(f"     - More parallelism = better CPU utilization")
    else:
        print(f"     - Current worker count is optimal")
    print()
    
    # Bottleneck analysis
    print("🔍 BOTTLENECK ANALYSIS")
    print("-" * 80)
    print("  Primary bottleneck: CPU (PDF extraction, text processing)")
    print("  Secondary bottleneck: Disk I/O (reading PDFs, writing JSON)")
    print()
    print("  How RAM helps:")
    print("  ✅ RAM disk → Faster I/O → Less CPU waiting")
    print("  ✅ Larger batches → More efficient CPU usage")
    print("  ✅ More workers (if RAM allows) → Better CPU utilization")
    print("  ⚠️  Models still loaded per-worker (Python multiprocessing limitation)")
    print()
    
    print("=" * 80)


if __name__ == '__main__':
    main()

