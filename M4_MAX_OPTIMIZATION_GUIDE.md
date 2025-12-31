# M4 Max Optimization Guide: Maximizing Performance with 128GB RAM

## 🎯 Current Status
- **Workers**: 20 (using ~50GB RAM)
- **Available RAM**: ~103GB (80% free)
- **GPU**: MPS enabled for embeddings (but embeddings disabled during processing)
- **CPU**: 16 cores (main bottleneck)

## 🚀 Additional Optimizations for M4 Max

### 1. **Increase Workers Further** ⭐ HIGH IMPACT

With 128GB RAM, you can safely use **24-28 workers**:

```yaml
processing:
  num_workers: 24  # or 28, test incrementally
```

**Why it works:**
- Each worker: ~2.5GB RAM
- 24 workers = ~60GB RAM (still 68GB free)
- 28 workers = ~70GB RAM (still 58GB free)
- More parallelism = better CPU utilization

**Test incrementally:**
- Try 24 workers first
- Monitor CPU usage (should stay <95%)
- If stable, try 26, then 28

**Expected improvement**: +10-15% additional speedup

---

### 2. **Enable GPU Acceleration for Chunking** ⭐ HIGH IMPACT

Currently, chunking uses CPU. Move it to GPU:

```yaml
chunking:
  method: "semantic"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "mps"  # ADD THIS - use GPU for chunking
```

**Why it works:**
- Chunking is CPU-intensive
- M4 Max GPU is powerful and underutilized
- Offloads work from CPU to GPU

**Expected improvement**: +15-25% speedup for chunking operations

---

### 3. **Use Async I/O for File Operations** ⭐ MEDIUM IMPACT

Replace blocking file I/O with async operations:

**Benefits:**
- CPU doesn't wait for disk I/O
- Workers can process while files are being read/written
- Better CPU utilization

**Implementation needed**: Add async file operations to worker functions

**Expected improvement**: +5-10% speedup

---

### 4. **Pre-load PDFs into Memory** ⭐ MEDIUM IMPACT

For frequently accessed PDFs, pre-load into RAM:

**Why it works:**
- Eliminates disk I/O wait time
- With 128GB RAM, can cache thousands of PDFs
- Massive speedup for repeated processing

**Implementation:**
- Create a PDF cache in RAM
- Pre-load next batch while processing current batch
- Use LRU cache for memory management

**Expected improvement**: +10-20% speedup (if PDFs are accessed multiple times)

---

### 5. **Optimize Model Loading** ⭐ MEDIUM IMPACT

**Current problem**: Each worker loads its own models (wastes CPU)

**Solutions:**

#### A. Use Threading for I/O-Bound Tasks
- Keep multiprocessing for CPU-bound tasks
- Use threading for I/O-bound tasks (file reading, writing)
- Reduces overhead

#### B. Model Quantization
- Use quantized models (smaller, faster)
- Less CPU per inference
- Slightly lower quality but much faster

```yaml
chunking:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Already small
  # Consider: "all-MiniLM-L12-v2" for better quality/speed balance
```

**Expected improvement**: +5-10% speedup

---

### 6. **Increase Batch Sizes Further** ⭐ LOW-MEDIUM IMPACT

You're already at 256 for embeddings. Increase other batch sizes:

```yaml
processing:
  batch_size: 2000  # Increase from 1000 (process more in memory)
  
# In memory_optimizer, increase:
embeddings:
  batch_size: 512  # Increase from 256 (if embeddings enabled)
```

**Why it works:**
- More data processed per CPU cycle
- Less overhead from batch management
- Better GPU utilization (if using GPU)

**Expected improvement**: +3-7% speedup

---

### 7. **Disable Unnecessary Features** ⭐ LOW IMPACT

Turn off features you don't need:

```yaml
pdf_extraction:
  enable_ocr: false  # If you don't need OCR (saves CPU)
  
text_processing:
  improve_formulas: false  # If formula formatting not critical
```

**Expected improvement**: +2-5% speedup (depending on usage)

---

### 8. **Use Core ML / Neural Engine** ⭐ EXPERIMENTAL

Apple's Neural Engine can accelerate ML operations:

**Benefits:**
- Dedicated ML acceleration
- Doesn't compete with CPU/GPU
- Very power efficient

**Limitations:**
- Requires Core ML model conversion
- Not all models supported
- More complex setup

**Expected improvement**: +10-20% for supported operations (if implemented)

---

### 9. **Optimize Database Operations** ⭐ LOW IMPACT

Batch database writes more aggressively:

```python
# Current: Write every 50 papers
batch_size_db = 50

# Optimized: Write every 200 papers
batch_size_db = 200
```

**Why it works:**
- Fewer database transactions
- Less CPU overhead
- Faster overall processing

**Expected improvement**: +2-3% speedup

---

### 10. **Parallel I/O Operations** ⭐ MEDIUM IMPACT

Read/write files in parallel:

**Implementation:**
- Use ThreadPoolExecutor for I/O operations
- Read next batch while processing current batch
- Write results asynchronously

**Expected improvement**: +5-10% speedup

---

## 📊 Priority Recommendations

### **Quick Wins** (Easy to implement, high impact):

1. ✅ **Increase workers to 24-28** (5 min config change)
2. ✅ **Enable GPU for chunking** (1 min config change)
3. ✅ **Increase batch_size to 2000** (1 min config change)
4. ✅ **Increase embedding batch_size to 512** (if embeddings enabled)

**Combined expected improvement**: +25-40% additional speedup

### **Medium Effort** (Requires code changes):

5. **Async I/O operations** (2-3 hours implementation)
6. **Pre-load PDFs into memory** (3-4 hours implementation)
7. **Parallel I/O operations** (2-3 hours implementation)

**Combined expected improvement**: +15-30% additional speedup

### **Advanced** (More complex):

8. **Core ML / Neural Engine** (1-2 days implementation)
9. **Model quantization** (requires testing)
10. **Hybrid threading/multiprocessing** (requires refactoring)

---

## 🎯 Recommended Configuration

```yaml
# Processing
processing:
  batch_size: 2000  # Increased from 1000
  num_workers: 24   # Increased from 20 (test, can go to 28)
  
# Chunking
chunking:
  method: "semantic"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "mps"  # NEW: Use GPU for chunking
  
# Embeddings (if enabled)
embeddings:
  batch_size: 512  # Increased from 256
  device: "mps"    # Already enabled
  
# Memory Optimization
memory_optimization:
  use_ram_disk: true
  ram_disk_size_gb: 30  # Increased from 20
  enable_model_caching: true
```

---

## 📈 Expected Combined Improvements

### Current (20 workers + memory optimization):
- **Time**: 18.15 min for 1000 papers
- **Throughput**: 3,290 papers/hour

### With Quick Wins (24 workers + GPU chunking + larger batches):
- **Expected time**: ~13-14 min for 1000 papers
- **Expected throughput**: ~4,200-4,600 papers/hour
- **Improvement**: +30-40% additional speedup

### Total Improvement from Baseline:
- **Original (12 workers)**: 27.59 min
- **Optimized (24 workers + all quick wins)**: ~13-14 min
- **Total improvement**: ~50% faster! 🚀

---

## ⚠️ Important Notes

1. **Monitor CPU Usage**: Should stay <95% for stability
2. **Test Incrementally**: Don't change everything at once
3. **GPU Memory**: M4 Max has unified memory, so GPU and CPU share RAM
4. **Thermal Throttling**: Monitor temperature, M4 Max may throttle if too hot

---

## 🧪 Testing Strategy

1. **Start with 24 workers** → Test → Monitor CPU
2. **Enable GPU for chunking** → Test → Verify speedup
3. **Increase batch sizes** → Test → Check memory usage
4. **If stable, try 26 workers** → Test → Monitor
5. **If still stable, try 28 workers** → Final test

**Stop if:**
- CPU usage >95% consistently
- System becomes unstable
- No further speedup observed

---

## 📝 Summary

With 128GB RAM and M4 Max, you have **massive headroom** for optimization:

- ✅ **More workers** (24-28 instead of 20)
- ✅ **GPU acceleration** for chunking (currently CPU-only)
- ✅ **Larger batches** (better CPU/GPU utilization)
- ✅ **Async I/O** (reduce CPU wait time)
- ✅ **Pre-loading** (eliminate I/O bottlenecks)

**Combined potential**: **50%+ total improvement** from original baseline!

