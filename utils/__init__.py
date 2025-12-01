# GPU acceleration with CuPy fallback to NumPy
import numpy as np

try:
    import cupy as cp
    # Test if GPU is actually accessible
    try:
        _ = cp.array([1, 2, 3])
        _ = cp.random.random(10)  # Test CURAND
        xp = cp
        GPU_AVAILABLE = True
        print("GPU (CuPy) enabled for acceleration")
    except Exception as e:
        print(f"GPU found but not accessible: {e}")
        print("Falling back to CPU (NumPy)")
        cp = np
        xp = np
        GPU_AVAILABLE = False
except ImportError:
    cp = np
    xp = np
    GPU_AVAILABLE = False
    print("GPU not available, using CPU (NumPy)")

def to_numpy(arr):
    """Convert CuPy array to NumPy if needed"""
    if GPU_AVAILABLE and hasattr(cp, 'asnumpy'):
        try:
            return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else arr
        except Exception:
            return np.asarray(arr)
    return np.asarray(arr) if not isinstance(arr, np.ndarray) else arr

def to_gpu(arr):
    """Move NumPy array to GPU if available"""
    if GPU_AVAILABLE and hasattr(cp, 'asarray'):
        try:
            return cp.asarray(arr) if not isinstance(arr, cp.ndarray) else arr
        except Exception:
            return arr
    return arr
