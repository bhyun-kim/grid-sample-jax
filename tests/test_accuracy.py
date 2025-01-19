import argparse
import datetime
import logging
import platform
import psutil  # for detailed CPU info (optional)
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import jax
import jax.numpy as jnp
import numpy as np

from PIL import Image
from io import BytesIO
import requests

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# Local import of your JAX grid_sample
from grid_sample_jax import grid_sample_jax

#############################################################################
# Configuration
#############################################################################

BATCH_SIZES = [4, 8, 16] 
# We'll only use the largest resolution for the 2D test:
RESOLUTION_2D = 128  # we skip smaller ones as per request
MODES = ["nearest", "bilinear", "bicubic"]
PADDING_MODES = ["zeros", "border", "reflection"]
ALIGN_CORNERS = [False, True]

def download_image(url="https://picsum.photos/640/480"):
    """
    Download an image from a URL and return it as a PIL Image (RGB).
    """
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img

def compute_mse_torch(t1, t2):
    """
    Compute MSE between two Tensors (PyTorch).
    """
    return ((t1 - t2) ** 2).mean().item()

def plot_comparison(original_tensor, torch_tensor, jax_tensor, 
                    mode, pad, ac, batch_idx=0, save_dir="comparison_figs"):
    """
    Creates a side-by-side comparison of the original image, the PyTorch result, and the JAX result.
    Saves it as a PNG file in 'save_dir'.

    original_tensor: [C, H, W] (the original input image for the batch_idx)
    torch_tensor:    [C, H, W] 
    jax_tensor:      [C, H, W] 
    mode: interpolation mode 
    pad:  padding mode
    ac:   align_corners
    batch_idx: which item in the batch
    """
    os.makedirs(save_dir, exist_ok=True)

    orig_pil  = to_pil_image(original_tensor)
    torch_pil = to_pil_image(torch_tensor)
    jax_pil   = to_pil_image(jax_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_pil)
    axes[0].set_title("Original")
    axes[1].imshow(torch_pil)
    axes[1].set_title("PyTorch")
    axes[2].imshow(jax_pil)
    axes[2].set_title("JAX")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"mode={mode}, pad={pad}, align_corners={ac}, batch_idx={batch_idx}")
    plt.tight_layout()

    filename = f"comparison_{mode}_{pad}_ac{ac}_batch{batch_idx}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

def test_2d_accuracy(device_choice):
    """
    Compare PyTorch and JAX grid_sample on 2D images for:
      - BATCH_SIZES
      - resolution = 128 (largest only)
      - modes in [nearest, bilinear, bicubic]
      - padding_modes in [zeros, border, reflection]
      - align_corners in [False, True]

    Generate a .log file with system info, date/time, 
    and all results. Also save side-by-side figures for each combo.
    """

    # ===========================
    #  Configure Logging
    # ===========================
    # Create a date-time string for the log file name
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/grid_sample_accuracy_{device_choice}_{now_str}.log"
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        filename=log_filename,
        filemode='w',  # overwrite if it already exists
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()

    # Also print logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # ===========================
    #  System & Library Info
    # ===========================
    cpu_info = platform.processor()
    if not cpu_info or cpu_info.strip() == '':
        # On some systems, platform.processor() can return an empty string.
        # Use psutil for more info, if installed:
        cpu_info = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads"
    gpu_info = "None"
    if device_choice == 'gpu' and torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)

    logger.info("==== Grid Sample Accuracy Test (2D) ====")
    logger.info(f"Date/Time: {now_str}")
    logger.info(f"Device choice: {device_choice}")
    logger.info(f"CPU info: {cpu_info}")
    logger.info(f"GPU info: {gpu_info}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"JAX version: {jax.__version__}")
    logger.info("")

    # ===========================
    #  Choose Device (PyTorch + JAX)
    # ===========================
    if device_choice == 'gpu' and torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')

    # JAX device
    if device_choice == 'gpu' and len(jax.devices("gpu")) > 0:
        jax_device = jax.devices("gpu")[0]
    else:
        jax_device = jax.devices("cpu")[0]

    logger.info(f"Selected PyTorch device: {torch_device}")
    logger.info(f"Selected JAX device: {jax_device.platform}")

    # ===========================
    #  Download & Prepare Data
    # ===========================
    base_pil = download_image()  # arbitrary image ~640x480
    logger.info("Downloaded base image from picsum.photos")

    transform = T.Compose([
        T.Resize((RESOLUTION_2D, RESOLUTION_2D)),
        T.ToTensor()
    ])
    single_img_tensor = transform(base_pil)  # [3, 128, 128] float
    logger.info(f"Single image shape: {single_img_tensor.shape}")

    # ===========================
    #  Test Loop
    # ===========================
    # We'll store aggregated MSE to compute an average.
    all_mse_values = []

    for batch_size in BATCH_SIZES:
        # Create a batch => [B, C, H, W]
        input_torch = single_img_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        input_torch = input_torch.to(torch_device)

        # Create an affine matrix for each item in batch => shape [B, 2, 3]
        angle_deg = 15
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        dx = 0.2  # normalized
        dy = -0.2

        affine_list = []
        for _ in range(batch_size):
            affine_list.append([
                [cos_a, -sin_a, dx],
                [sin_a,  cos_a, dy]
            ])
        affine_mtx = torch.tensor(affine_list, dtype=torch.float32, device=torch_device)

        # Make a grid => shape [B, H, W, 2]
        # Use align_corners=False for grid creation only
        grid_torch = F.affine_grid(
            affine_mtx, size=input_torch.size(),
            align_corners=False
        )

        # Convert input to JAX => NHWC
        input_jax = jnp.array(input_torch.permute(0,2,3,1).cpu().numpy())
        grid_jax  = jnp.array(grid_torch.cpu().numpy())

        # Move to JAX device
        input_jax = jax.device_put(input_jax, jax_device)
        grid_jax  = jax.device_put(grid_jax,  jax_device)

        # Original (to show in figure) => take first item in batch
        original_first = input_torch[0].cpu()  # shape [3,128,128]

        for mode in MODES:
            for pad in PADDING_MODES:
                for ac in ALIGN_CORNERS:
                    # PyTorch => shape [B,C,H,W]
                    torch_out = F.grid_sample(
                        input_torch, grid_torch,
                        mode=mode,
                        padding_mode=pad,
                        align_corners=ac
                    )

                    # JAX => shape [B,H,W,C]
                    jax_out_nhwc = grid_sample_jax(
                        input_jax, grid_jax,
                        mode=mode,
                        padding_mode=pad,
                        align_corners=ac
                    )
                    # Convert back to [B,C,H,W] for comparison
                    jax_out = torch.from_numpy(np.array(jax_out_nhwc)).permute(0,3,1,2)
                    jax_out = jax_out.to(torch_device)

                    # MSE
                    mse_value = compute_mse_torch(torch_out, jax_out)
                    all_mse_values.append(mse_value)

                    # Log the result
                    msg = (f"2D -> B={batch_size}, Res={RESOLUTION_2D}, "
                           f"Mode={mode}, Pad={pad}, AC={ac} | MSE={mse_value:.6f}")
                    logger.info(msg)

                    # For figures, we only need to save one from the batch (e.g. index 0):
                    # Plot original, torch_out[0], jax_out[0]
                    plot_comparison(
                        original_first,
                        torch_out[0].cpu(),
                        jax_out[0].cpu(),
                        mode=mode, pad=pad, ac=ac,
                        batch_idx=batch_size  # or just use 0
                    )

    # ===========================
    #  Average MSE
    # ===========================
    if len(all_mse_values) > 0:
        avg_mse = sum(all_mse_values) / len(all_mse_values)
    else:
        avg_mse = 0.0

    logger.info("")
    logger.info(f"=== Finished all tests. Average MSE = {avg_mse:.6f} ===")
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

    test_2d_accuracy(args.device)
