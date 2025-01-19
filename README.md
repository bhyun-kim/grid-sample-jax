# Grid-Sample-JAX

This repository provides a **JAX** reimplementation of PyTorch's `torch.nn.functional.grid_sample`, along with a **test.py** to compare results between PyTorch and JAX. It includes:

---

## Installation

### Install via Git

If you just want to install directly from the GitHub repository, you can run:

```bash
pip install git+https://github.com/bhyun-kim/grid-sample-jax.git
```

---

## Usage

### Running the End-to-End Tests

1. Ensure you have installed the required dependencies (via either of the methods above).
2. Run the test scripts from the `tests` folder:

   ```bash
   python tests/test_accuracy.py --device <cpu|gpu>
   python tests/test_torch_speed.py --device <cpu|gpu>
   python tests/test_jax_speed.py --device <cpu|gpu>
   ```

   This will:

   - For `test_accuracy.py`:
     - Download a random image from [Picsum](https://picsum.photos/)
     - Generate multiple batches at different resolutions
     - Apply affine transformations and run both **PyTorch** and **JAX** grid sampling
     - Compute **MSE** differences and store results in log files

   - For `test_torch_speed.py`:
     - Measure the speed of **PyTorch** grid sampling
     - Store speed results in log files

   - For `test_jax_speed.py`:
     - Measure the speed of **JAX** grid sampling
     - Store speed results in log files

## Discussion 

### Accuracy Results

The JAX implementation shows near-identical results compared to PyTorch's `grid_sample` across all configurations. From the GPU tests:

- Average MSE across all configurations: 0.000003
- Maximum observed MSE: 0.000020 (with nearest mode, reflection padding)
- Most configurations achieve MSE of 0.000000

These results demonstrate that this JAX implementation produces numerically equivalent results to PyTorch's implementation across all interpolation modes (nearest, bilinear, bicubic), padding modes (zeros, border, reflection), and align_corners settings.

### Performance Analysis

Performance comparison shows the current state of implementation between JAX and PyTorch. Note that the JAX implementation is still in development and has room for optimization in terms of both compilation time and execution speed.

#### CPU Performance

| Configuration | JAX (steady state) | PyTorch | Notes |
|--------------|-------------------|----------|--------|
| B=32, Res=64, nearest | ~0.000010s | ~0.000087s | JAX shows better performance |
| B=32, Res=64, bilinear | ~0.000009s | ~0.000220s | JAX maintains efficiency |
| B=32, Res=64, bicubic | ~0.000014s | ~0.000957s | JAX significantly faster |

#### GPU Performance (RTX 3050)

| Configuration | JAX (steady state) | PyTorch | Notes |
|--------------|-------------------|----------|--------|
| B=32, Res=128, bilinear | ~0.000156s | ~0.000029s | PyTorch shows ~5x faster execution |
| B=128, Res=256, bicubic | ~0.000123s | ~0.000201s | JAX performs better at larger batch sizes |

Observations:
1. CPU Performance: JAX implementation generally shows better performance on CPU
2. GPU Performance: PyTorch shows advantages for smaller batch sizes, while JAX becomes competitive at larger scales
3. Future work is needed to optimized the overall performance

---
## Future Work

### 3D Implementation
- [ ] Implement 3D grid sampling core functionality
- [ ] Create test suite for 3D operations
- [ ] Validate against PyTorch's 3D grid_sample

### Code Optimization
- [ ] Profile and identify performance bottlenecks
- [ ] Optimize GPU performance:

---

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or improvements, please let me know.

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use and modify it, but please retain the license and credit.