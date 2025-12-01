# GPU acceleration with CuPy fallback to NumPy
import numpy as np
import os

# Force specific GPU device if set
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    import cupy as cp
    # Test if GPU is actually accessible
    try:
        # Set memory pool to avoid fragmentation
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        
        # Test basic operations
        test_arr = cp.array([1, 2, 3])
        
        # For older GPUs (compute capability < 6.0), use legacy random
        device = cp.cuda.Device(0)
        cc_major, cc_minor = device.compute_capability
        print(f"GPU Compute Capability: {cc_major}.{cc_minor}")
        
        # Convert to int if string
        if isinstance(cc_major, str):
            cc_major = int(cc_major.split('_')[0]) if '_' in cc_major else int(cc_major)
        
        if cc_major < 6:
            print("Older GPU detected, using NumPy random with CuPy arrays")
            # Use NumPy for random, CuPy for array ops
            import numpy.random as cp_random
            cp.random = cp_random
        else:
            # Test CURAND for newer GPUs
            _ = cp.random.random(10)
        
        xp = cp
        GPU_AVAILABLE = True
        print(f"GPU (CuPy) enabled for acceleration on {device.name.decode()}")
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
