# GPU acceleration with CuPy fallback to NumPy
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) enabled for acceleration")
except ImportError:
    import numpy as cp
    xp = cp
    GPU_AVAILABLE = False
    print("GPU not available, using CPU (NumPy)")

def to_numpy(arr):
    """Convert CuPy array to NumPy if needed"""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def to_gpu(arr):
    """Move NumPy array to GPU if available"""
    if GPU_AVAILABLE and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    return arr
