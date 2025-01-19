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
2. Run the main test script from the project root (depending on your structure):

   ```bash
   python test.py
   ```

   This will:
   - Download a random image from [Picsum](https://picsum.photos/)
   - Generate multiple batches at different resolutions
   - Apply affine transformations and run both **PyTorch** and **JAX** grid sampling
   - Compute **MSE** differences, measure speed, and store results in a local SQLite database
   - Save a comparison image (`comparison_example.png`) for one selected configuration

### Examining the SQLite Database

After the tests complete, you will find a file named (for example) `grid_sample_results.db`. You can inspect the results with:

```bash
sqlite3 grid_sample_results.db
```

This should show you rows for each combination of:
- Batch size
- Resolution
- Mode
- Padding mode
- Align corners

Each row also lists the **MSE**, **torch_time**, and **jax_time**.

---

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or improvements, please let me know.

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use and modify it, but please retain the license and credit.