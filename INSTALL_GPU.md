# Cài đặt GPU Support cho Better-FIM

Better-FIM giờ hỗ trợ GPU acceleration thông qua CuPy với automatic fallback về NumPy nếu không có GPU.

## Yêu cầu

- NVIDIA GPU với CUDA support
- CUDA Toolkit 11.x hoặc 12.x
- Python 3.8+

## Cài đặt CuPy

### Option 1: Cài đặt tự động (khuyến nghị)

```bash
pip install cupy-cuda11x  # Cho CUDA 11.x
# hoặc
pip install cupy-cuda12x  # Cho CUDA 12.x
```

### Option 2: Kiểm tra CUDA version trước

```bash
nvcc --version  # Kiểm tra CUDA version
```

Sau đó cài CuPy phù hợp:
- CUDA 11.2-11.8: `pip install cupy-cuda11x`
- CUDA 12.x: `pip install cupy-cuda12x`

### Option 3: Không có GPU

Nếu không có GPU, Better-FIM tự động dùng NumPy (CPU mode):

```bash
# Không cần làm gì, code tự detect
python main.py  # Sẽ hiện "GPU not available, using CPU (NumPy)"
```

## Kiểm tra GPU hoạt động

```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Số lượng GPU
print(cp.cuda.Device(0).compute_capability)  # GPU capability
```

## Lợi ích khi dùng GPU

| Thao tác | CPU (NumPy) | GPU (CuPy) | Tăng tốc |
|----------|-------------|------------|----------|
| Bernoulli sampling (IC) | Chậm | Nhanh | ~3-5x |
| Array sorting (fitness) | Chậm | Nhanh | ~2-3x |
| Probability normalization | Chậm | Nhanh | ~4-6x |
| Community weighting | Chậm | Nhanh | ~3-4x |

**Lưu ý:** Tốc độ tăng phụ thuộc vào:
- Kích thước graph (>1000 nodes mới thấy rõ)
- Số MC simulations (mc >= 500)
- GPU memory và compute capability

## Troubleshooting

### "No module named 'cupy'"
```bash
pip install cupy-cuda11x  # hoặc cuda12x
```

### "CUDA_ERROR_NO_DEVICE"
- Kiểm tra NVIDIA driver: `nvidia-smi`
- Cài đặt/update CUDA Toolkit

### Memory errors
Giảm `MC_SIMULATIONS` trong `betterFIM.py`:
```python
MC_SIMULATIONS = 500  # Thay vì 1000
```

## Benchmark

Chạy test để so sánh CPU vs GPU:

```bash
cd Better-FIM
python -c "from utils import GPU_AVAILABLE; print('GPU:', GPU_AVAILABLE)"
time python main.py  # Đo thời gian chạy
```

## Tắt GPU (force CPU mode)

Nếu muốn force dùng CPU dù có GPU:

```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
python main.py
```

Hoặc edit `utils/__init__.py`:
```python
GPU_AVAILABLE = False  # Force CPU
```
