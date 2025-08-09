import torch
import gc
from typing import Optional

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not installed. Memory monitoring disabled.")

# Memory threshold (can be overridden from config)
DEFAULT_MEMORY_THRESHOLD_MB = 8000


def monitor_memory_usage() -> float:
    """Return current process RSS in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb
    except Exception as e:
        print(f"⚠️ Memory monitoring error: {e}")
        return 0.0


def check_memory_limit(threshold_mb: Optional[float] = None) -> bool:
    """Check whether current RSS exceeds threshold_mb."""
    if threshold_mb is None:
        threshold_mb = DEFAULT_MEMORY_THRESHOLD_MB
        
    current_memory = monitor_memory_usage()
    if current_memory > threshold_mb:
        print(f"⚠️ High memory usage: {current_memory:.1f}MB (threshold: {threshold_mb}MB)")
        return True
    return False


def safe_memory_cleanup() -> None:
    """Best-effort memory cleanup for CUDA and Python GC."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        print(f"⚠️ Memory cleanup error: {e}")


def log_memory_status(prefix: str = "") -> None:
    """Log current memory usage; include CUDA stats when available."""
    memory_mb = monitor_memory_usage()
    if memory_mb > 0:
        print(f"📊 {prefix}memory usage: {memory_mb:.1f}MB")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2     # MB
        print(f"🎮 CUDA memory: {allocated:.1f}MB allocated, {cached:.1f}MB reserved")


def memory_efficient_batch_processing(data_list, batch_size: int = 1000, 
                                     memory_check_interval: int = 10):
    """Generator that yields batches with periodic memory checks."""
    for i in range(0, len(data_list), batch_size):
        if i // batch_size % memory_check_interval == 0:
            if check_memory_limit():
                print(f"⚠️ Reducing batch size due to memory pressure: {batch_size} → {batch_size//2}")
                batch_size = max(batch_size // 2, 100)
        
        batch = data_list[i:i+batch_size]
        yield batch
        
        if i % (batch_size * 5) == 0:
            safe_memory_cleanup()