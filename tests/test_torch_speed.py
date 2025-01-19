#!/usr/bin/env python

import time
import sqlite3
import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import argparse
import logging
import datetime
import platform
import psutil  # for CPU info

#############################################################################
# Configuration
#############################################################################

BATCH_SIZES = [32, 64, 128]
RESOLUTIONS = [64, 128, 256]
MODES = ["nearest", "bilinear", "bicubic"]
PADDING_MODES = ["zeros", "border", "reflection"]
ALIGN_CORNERS = [False, True]

NUM_RUNS = 5  # how many runs we measure for the average timing

def download_image(url="https://picsum.photos/640/480"):
    """
    Download an image from a URL and return it as a PIL Image (RGB).
    """
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def make_random_affine_matrix(batch_size, resolution, dx=20, dy=10, angle_deg=15):
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
    return torch.tensor(matrix, dtype=torch.float32)


def measure_time_torch(input_torch, grid_torch, mode, pad, align_corners,
                       device, runs=NUM_RUNS):
    """
    Measure PyTorch grid_sample time.
    We'll do:
      - a couple warmup calls
      - measure the average time over 'runs' calls
    """
    # Warmup (e.g. 2 calls)
    for _ in range(2):
        _ = F.grid_sample(input_torch, grid_torch, mode=mode,
                          padding_mode=pad, align_corners=align_corners)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(runs):
        _ = F.grid_sample(input_torch, grid_torch, mode=mode,
                          padding_mode=pad, align_corners=align_corners)
    end = time.perf_counter()

    return (end - start) / runs

def main(device_choice):
    # ------------------------------
    # Logging setup
    # ------------------------------
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"torch_speed_test_{device_choice}_{now_str}.log"

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
    # System / Library Info
    # ------------------------------
    cpu_info = platform.processor()
    if not cpu_info.strip():
        cpu_info = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads"
    gpu_info = "None"
    if device_choice == 'gpu' and torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
    
    logger.info("==== PyTorch Speed Test ====")
    logger.info(f"Date/Time: {now_str}")
    logger.info(f"Device choice: {device_choice}")
    logger.info(f"CPU info: {cpu_info}")
    logger.info(f"GPU info: {gpu_info}")
    logger.info(f"PyTorch version: {torch.__version__}\n")

    # ------------------------------
    # Torch device selection
    # ------------------------------
    if device_choice == 'gpu' and torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')
    logger.info(f"Selected PyTorch device: {torch_device}\n")

    # ------------------------------
    # Download image
    # ------------------------------
    base_img = download_image()
    logger.info("Image downloaded from picsum.photos")

    # We'll collect times to compute an average steady-state time
    all_times = []

    # ------------------------------
    # Main loop over combos
    # ------------------------------
    for batch_size in BATCH_SIZES:
        for resolution in RESOLUTIONS:
            for mode in MODES:
                for pad in PADDING_MODES:
                    for ac in ALIGN_CORNERS:
                        
                        base_resized = base_img.resize((resolution, resolution), Image.BILINEAR)
                        base_np = np.array(base_resized, dtype=np.float32) / 255.0
                        # shape => [res, res, 3], convert to torch => [3, res, res]
                        base_tensor = torch.from_numpy(base_np).permute(2,0,1)
                        # replicate => [B, 3, H, W]
                        input_torch = base_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                        input_torch = input_torch.to(torch_device)

                        affine_mtx = make_random_affine_matrix(batch_size, resolution).to(torch_device)
                        grid_torch = F.affine_grid(
                            affine_mtx,
                            size=input_torch.size(),  # (B,3,H,W)
                            align_corners=False
                        )

                        time_avg = measure_time_torch(
                            input_torch, grid_torch,
                            mode=mode, pad=pad, align_corners=ac,
                            device=torch_device, runs=NUM_RUNS
                        )
                        all_times.append(time_avg)

                        msg = (f"B={batch_size}, Res={resolution}, mode={mode}, pad={pad}, "
                               f"align_corners={ac} => time={time_avg:.6f}s")
                        logger.info(msg)

    # ------------------------------
    # Final summary
    # ------------------------------
    if all_times:
        avg_time = sum(all_times) / len(all_times)
    else:
        avg_time = 0.0

    logger.info("")
    logger.info(f"=== Finished all tests. Average time = {avg_time:.6f} s ===")
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
