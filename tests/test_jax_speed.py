#!/usr/bin/env python

import time
import sqlite3
import requests
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from io import BytesIO
import argparse
import logging
import datetime
import platform
import psutil   # for CPU info (optional)
import torch    # just for version info, if desired

# Your local module with the JAX grid_sample function:
# Make sure it supports (mode, padding_mode, align_corners) arguments
from grid_sample_jax import grid_sample_jax

#############################################################################
# Configuration
#############################################################################

BATCH_SIZES = [32, 64, 128]
RESOLUTIONS = [64, 128, 256]
MODES = ["nearest", "bilinear", "bicubic"]
PADDING_MODES = ["zeros", "border", "reflection"]
ALIGN_CORNERS = [False, True]

NUM_RUNS = 5  # how many runs for steady-state timing

def download_image(url="https://picsum.photos/640/480"):
    """
    Download an image from a URL and return it as a PIL Image (RGB).
    """
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def make_random_affine_matrix_jax(batch_size, resolution, dx=20, dy=10, angle_deg=15):
    """
    Create a simple 2D affine matrix for each item in the batch (rotation+translation).
    Returns shape (B, 2, 3).
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    H, W = resolution, resolution
    tx = dx / (W / 2.0)  # normalized translation
    ty = dy / (H / 2.0)

    matrix = []
    for _ in range(batch_size):
        matrix.append([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty]
        ])
    return np.array(matrix, dtype=np.float32)

def affine_grid_jax(affine_mtx, out_shape, align_corners=False):
    """
    Builds a grid [B, H, W, 2] for each batch using the given affine matrices.

    out_shape is (B, C, H, W); we only need (H, W) for grid sampling.
    """
    B = out_shape[0]
    _, _, H, W = out_shape

    xs = np.linspace(-1, 1, W, dtype=np.float32)
    ys = np.linspace(-1, 1, H, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs)  # shape [H, W]
    base_grid = np.stack([grid_x, grid_y], axis=-1)  # [H, W, 2]
    base_grid = np.tile(base_grid[None, ...], [B, 1, 1, 1])  # => [B, H, W, 2]

    # Add homogeneous coordinate of 1 => shape [B, H, W, 3]
    ones = np.ones((B, H, W, 1), dtype=np.float32)
    base_grid_homo = np.concatenate([base_grid, ones], axis=-1)

    out_grid = []
    for b in range(B):
        mat = affine_mtx[b]  # [2, 3]
        coords_b = base_grid_homo[b].reshape(-1, 3)   # [H*W, 3]
        coords_out = coords_b @ mat.T                # => [H*W, 2]
        coords_out = coords_out.reshape(H, W, 2)
        out_grid.append(coords_out)

    return np.stack(out_grid, axis=0).astype(np.float32)  # => [B, H, W, 2]

def measure_time_jax(
    input_jax,
    grid_jax,
    mode,
    padding_mode,
    align_corners,
    runs=NUM_RUNS
):
    """
    Measure JAX grid_sample time for the specified (mode, padding_mode, align_corners).
    We do two phases:
      1) Compilation/Warmup (first call).
      2) Steady-state average time (subsequent calls).

    Returns: (compile_time, steady_state_time)
    """

    def call_fn(x, g):
        return grid_sample_jax(
            x, g,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners
        )

    fn = jax.jit(call_fn)

    # 1) Compilation time
    start_compile = time.perf_counter()
    _ = fn(input_jax, grid_jax)  # triggers JIT compilation
    end_compile = time.perf_counter()
    compile_time = end_compile - start_compile

    # Additional warmup call
    _ = fn(input_jax, grid_jax)

    # 2) Steady-state measurement
    start_infer = time.perf_counter()
    for _ in range(runs):
        _ = fn(input_jax, grid_jax)
    end_infer = time.perf_counter()

    steady_state_time = (end_infer - start_infer) / runs
    return compile_time, steady_state_time

def main(device_choice):
    # ------------------------------
    #  Logging setup
    # ------------------------------
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"jax_speed_test_{device_choice}_{now_str}.log"

    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # ------------------------------
    #  System / Library Info
    # ------------------------------
    cpu_info = platform.processor()
    if not cpu_info.strip():
        cpu_info = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads"
    gpu_info = "None"
    if device_choice == 'gpu' and len(jax.devices("gpu")) > 0:
        gpu_info = jax.devices("gpu")[0].device_kind
    
    logger.info("==== JAX Speed Test ====")
    logger.info(f"Date/Time: {now_str}")
    logger.info(f"Device choice: {device_choice}")
    logger.info(f"CPU info: {cpu_info}")
    logger.info(f"GPU info: {gpu_info}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"JAX version: {jax.__version__}\n")

    # ------------------------------
    #  JAX device selection
    # ------------------------------
    if device_choice == 'gpu' and len(jax.devices("gpu")) > 0:
        jax_device = jax.devices("gpu")[0]
    else:
        jax_device = jax.devices("cpu")[0]

    logger.info(f"Selected JAX device: {jax_device.platform}\n")

    # ------------------------------
    #  Download or prepare image
    # ------------------------------
    base_img = download_image()
    logger.info("Image downloaded from picsum.photos")

    # We'll collect times to compute an overall average steady-state time
    all_steady_times = []

    # ------------------------------
    #  Main loop over combos
    # ------------------------------
    for batch_size in BATCH_SIZES:
        for resolution in RESOLUTIONS:
            for mode in MODES:
                for pad in PADDING_MODES:
                    for ac in ALIGN_CORNERS:
                        # Prepare input => [B, H, W, C]
                        base_resized = base_img.resize((resolution, resolution), Image.BILINEAR)
                        base_np = np.array(base_resized, dtype=np.float32) / 255.0
                        # replicate for batch => shape (B, H, W, C)
                        input_np = np.stack([base_np]*batch_size, axis=0)

                        # Move to JAX device
                        input_jax = jnp.array(input_np)
                        input_jax = jax.device_put(input_jax, jax_device)

                        # Build affine matrix & grid => [B, H, W, 2]
                        affine_mtx = make_random_affine_matrix_jax(batch_size, resolution)
                        grid_np = affine_grid_jax(
                            affine_mtx,
                            out_shape=(batch_size, 3, resolution, resolution),
                            align_corners=ac
                        )
                        grid_jax_arr = jax.device_put(jnp.array(grid_np), jax_device)

                        # Time measurement (with the specified mode/padding/align)
                        compile_time, steady_time = measure_time_jax(
                            input_jax,
                            grid_jax_arr,
                            mode=mode,
                            padding_mode=pad,
                            align_corners=ac,
                            runs=NUM_RUNS
                        )
                        all_steady_times.append(steady_time)


                        msg = (f"B={batch_size}, Res={resolution}, mode={mode}, pad={pad}, "
                               f"align_corners={ac} => compile_time={compile_time:.6f}s, "
                               f"steady_time={steady_time:.6f}s")
                        logger.info(msg)

    # ------------------------------
    #  Summary
    # ------------------------------
    if all_steady_times:
        avg_steady = sum(all_steady_times) / len(all_steady_times)
    else:
        avg_steady = 0.0

    logger.info("")
    logger.info(f"=== Finished all tests. Average steady_time = {avg_steady:.6f} s ===")
    logger.info(f"Log file saved to: {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help='Choose whether to run on CPU or GPU.'
    )
    args = parser.parse_args()
    main(args.device)
