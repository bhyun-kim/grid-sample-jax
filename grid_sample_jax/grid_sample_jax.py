import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def grid_sample_jax(
    img: jnp.ndarray,
    grid: jnp.ndarray,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    cubic_A: float = -0.75,
) -> jnp.ndarray:
    """
    A unified 2D grid sampling function in JAX, supporting:
      - mode:          ['nearest', 'bilinear', 'bicubic']
      - padding_mode:  ['zeros', 'border', 'reflection']
      - align_corners: bool
      - cubic_A:       float (parameter for bicubic)

    Args:
      img:    2D => (N, H_in, W_in, C)
      grid:   2D => (N, H_out, W_out, 2) with coords in [-1,1]
      mode:   'nearest', 'bilinear', or 'bicubic'
      padding_mode: 'zeros', 'border', or 'reflection'
      align_corners: see PyTorch grid_sample docs for definition
      cubic_A:   The "A" parameter used in the bicubic kernel.

    Returns:
      Sampled image of shape:
        2D => (N, H_out, W_out, C)

    3D grid sampling is not yet supported.
    """
    num_dims = grid.shape[-1]  # either 2 (2D) or 3 (3D)
    batch_size = img.shape[0]
    
    if num_dims == 2:
        # ---------------------
        # 2D Grid Sample Logic
        # ---------------------
        N, H_in, W_in, C = img.shape
        assert N == batch_size, "img and grid must have same N (batch dim)."

        # grid => (N, H_out, W_out, 2)
        H_out = grid.shape[1]
        W_out = grid.shape[2]

        # 1) Unnormalize: get float coords in image space
        x_f, y_f = _unnormalize_grid_2d(grid, H_in, W_in, align_corners)

        # 2) Handle interpolation modes
        if mode == "nearest":
            # nearest-neighbor
            x_idx, y_idx = _compute_nearest_indices_2d(x_f, y_f)
            
            # Gather from these integer coords (K=1 corner)
            corners = _gather_2d(img, x_idx[..., None], y_idx[..., None], padding_mode)
            # shape => (N, H_out, W_out, 1, C)
            out = corners[..., 0, :]  # => (N, H_out, W_out, C)

        elif mode == "bilinear":
            # bilinear => need (x0, x1, y0, y1, wx0, wx1, wy0, wy1)
            x0, x1, y0, y1, wx0, wx1, wy0, wy1 = _compute_bilinear_indices_weights_2d(x_f, y_f)

            # stack corners for gather => shape (N,H_out,W_out,4)
            x_idx = jnp.stack([x0, x1, x0, x1], axis=-1)
            y_idx = jnp.stack([y0, y0, y1, y1], axis=-1)

            # gather => (N, H_out, W_out, 4, C)
            corners = _gather_2d(img, x_idx, y_idx, padding_mode)

            # weights => (N, H_out, W_out, 4)
            w = jnp.stack([
                wx0 * wy0,  # top-left
                wx1 * wy0,  # top-right
                wx0 * wy1,  # bottom-left
                wx1 * wy1,  # bottom-right
            ], axis=-1)

            # Weighted sum => (N, H_out, W_out, C)
            out = jnp.sum(corners * w[..., None], axis=-2)

        elif mode == "bicubic":
            # bicubic => gather 4x4 in x and y
            x_idx, y_idx, w_x, w_y = _compute_bicubic_indices_weights_2d(x_f, y_f, cubic_A)

            # For bicubic, we do a separable approach: gather 4 corners in x, 4 corners in y
            # Actually we gather 16 points total: all pairs (x_idx[:,i], y_idx[:,j]) for i,j in [0..3].
            # One common approach: gather in two passes or do it all at once. Here we do all at once.

            # We'll replicate x_idx,y_idx so each entry covers all combos of x0..x3, y0..y3
            # shape => (N, H_out, W_out, 4, 4) => flatten to 16
            # But simpler is to gather in one pass: we create the outer product of x_idx and y_idx, then gather.

            # Let's do a broadcast approach:
            # Expand dims so we can combine (x0..x3) with (y0..y3)
            # x_idx => (N, H_out, W_out, 4)
            # y_idx => (N, H_out, W_out, 4)

            # We want to gather all 16 combos => (N, H_out, W_out, 16).
            # We'll flatten that last dimension to do a single gather_2d call.

            x_idx_b = x_idx[..., None].repeat(4, axis=-1)  # shape => (N,H_out,W_out,4*4=16)
            y_idx_b = y_idx[..., None, :].repeat(4, axis=-2)  # also => 16
            x_idx_b = x_idx_b.reshape(x_idx_b.shape[:3] + (16,))
            y_idx_b = y_idx_b.reshape(y_idx_b.shape[:3] + (16,))

            # gather => (N, H_out, W_out, 16, C)
            corners = _gather_2d(img, x_idx_b, y_idx_b, padding_mode)

            # Now we need weights for each of the 16 corners. 
            # w_x => (N,H_out,W_out,4), w_y => (N,H_out,W_out,4).
            # We'll do an outer product: w_x_i * w_y_j => shape (N,H_out,W_out,4,4) => flatten to 16
            w_x_b = w_x[..., None]  # (N,H_out,W_out,4,1)
            w_y_b = w_y[..., None, :]  # (N,H_out,W_out,1,4)
            w_xy = w_x_b * w_y_b  # => (N,H_out,W_out,4,4)
            w_xy = w_xy.reshape(w_xy.shape[:3] + (16,))  # => (N,H_out,W_out,16)

            # Weighted sum => (N,H_out,W_out,C)
            out = jnp.sum(corners * w_xy[..., None], axis=-2)

        else:
            raise ValueError(f"Unknown mode='{mode}' for 2D")

        return out

    elif num_dims == 3:
        # ---------------------
        # 3D Grid Sample Logic (not implemented yet)
        # ---------------------
        raise NotImplementedError("3D grid sampling is not yet supported. Please use a 2D input.")

    else:
        raise ValueError(
            f"grid_sample_jax: last dim of grid must be 2 (2D) or 3 (3D), got {num_dims}."
        )


def _reflect_index_vectorized(
    idx: jnp.ndarray,
    size: int
) -> jnp.ndarray:
    """
    Reflect out-of-bounds indices into [0, size-1].
    Example for size=5: valid [0..4].
      -1 -> 1,  -2 -> 2, 5 -> 3, etc.

    idx shape can be (N, ..., K). 
    """
    if size <= 1:
        return jnp.zeros_like(idx, dtype=jnp.int32)

    double_range = 2 * (size - 1)
    idx_mod = jnp.abs(idx) % double_range
    reflected = jnp.where(
        idx_mod >= (size - 1),
        double_range - idx_mod,
        idx_mod
    )
    return reflected.astype(jnp.int32)

def _clip_index_vectorized(
    idx: jnp.ndarray,
    size: int
) -> jnp.ndarray:
    """
    Clamp out-of-bounds indices into [0, size-1].
    idx shape: (N, ..., K)
    """
    return jnp.clip(idx, 0, size - 1)

def _cubic_kernel_weights(
    t: jnp.ndarray,
    A: float = -0.75
) -> jnp.ndarray:
    """
    1D bicubic kernel for distance `t`.
    Common A: -0.75 (Mitchell-Netravali) or -0.5 for B-spline-like interpolation.
    """
    t = jnp.abs(t)
    t2 = t * t
    t3 = t2 * t

    w = ((A + 2.0) * t3 - (A + 3.0) * t2 + 1.0) * (t <= 1.0) \
        + (A * t3 - 5.0*A * t2 + 8.0*A * t - 4.0*A) * ((t > 1.0) & (t < 2.0))
    return w

def _unnormalize_grid_2d(
    grid: jnp.ndarray,
    H_in: int,
    W_in: int,
    align_corners: bool
):
    """
    Convert normalized coords (x_norm, y_norm) in [-1, 1] 
    to absolute coords (x_f, y_f) in image space.

    grid shape: (N, H_out, W_out, 2)

    If align_corners=True:
      x_norm in [-1,1] -> x_f in [0,   W_in - 1]
      y_norm in [-1,1] -> y_f in [0,   H_in - 1]

    If align_corners=False:
      x_norm in [-1,1] -> x_f in [-0.5,   W_in-0.5]
      y_norm in [-1,1] -> y_f in [-0.5,   H_in-0.5]
    """
    x_norm = grid[..., 0]
    y_norm = grid[..., 1]

    if align_corners:
        x_f = 0.5 * (x_norm + 1.0) * (W_in - 1)
        y_f = 0.5 * (y_norm + 1.0) * (H_in - 1)
    else:
        x_f = (x_norm + 1.0) * 0.5 * W_in - 0.5
        y_f = (y_norm + 1.0) * 0.5 * H_in - 0.5

    return x_f, y_f

def _gather_2d_zeros(
    img: jnp.ndarray,
    x_idx: jnp.ndarray,
    y_idx: jnp.ndarray
) -> jnp.ndarray:
    """
    2D zeros padding gather.
    Out-of-bounds => 0.

    img: (N, H_in, W_in, C)
    x_idx,y_idx: (N, H_out, W_out, K)
    Return: (N, H_out, W_out, K, C)
    """
    N, H_out, W_out, K = x_idx.shape
    _, H_in, W_in, C = img.shape

    x_clamped = jnp.clip(x_idx, 0, W_in - 1)
    y_clamped = jnp.clip(y_idx, 0, H_in - 1)

    b_idx = jnp.arange(N)[:, None, None, None]
    b_idx = jnp.broadcast_to(b_idx, x_idx.shape)

    gathered = img[b_idx, y_clamped, x_clamped, :]  # (N,H_out,W_out,K,C)

    # Zero out where OOB
    oob_mask = (
        (x_idx < 0) | (x_idx >= W_in) |
        (y_idx < 0) | (y_idx >= H_in)
    )[..., None]

    return jnp.where(oob_mask, 0.0, gathered)

def _gather_2d_border(
    img: jnp.ndarray,
    x_idx: jnp.ndarray,
    y_idx: jnp.ndarray
) -> jnp.ndarray:
    """
    2D border (clamp) gather.
    """
    _, H_in, W_in, C = img.shape

    x_clamped = _clip_index_vectorized(x_idx, W_in)
    y_clamped = _clip_index_vectorized(y_idx, H_in)

    b_idx = jnp.arange(img.shape[0])[:, None, None, None]
    b_idx = jnp.broadcast_to(b_idx, x_idx.shape)

    return img[b_idx, y_clamped, x_clamped, :]

def _gather_2d_reflect(
    img: jnp.ndarray,
    x_idx: jnp.ndarray,
    y_idx: jnp.ndarray
) -> jnp.ndarray:
    """
    2D reflection gather.
    """
    _, H_in, W_in, C = img.shape

    x_reflect = _reflect_index_vectorized(x_idx, W_in)
    y_reflect = _reflect_index_vectorized(y_idx, H_in)

    b_idx = jnp.arange(img.shape[0])[:, None, None, None]
    b_idx = jnp.broadcast_to(b_idx, x_idx.shape)

    return img[b_idx, y_reflect, x_reflect, :]

def _gather_2d(
    img: jnp.ndarray,
    x_idx: jnp.ndarray,
    y_idx: jnp.ndarray,
    padding_mode: str = "zeros"
) -> jnp.ndarray:
    """
    Switch for the chosen 2D padding_mode.
    Returns: (N, H_out, W_out, K, C)
    """
    if padding_mode == "reflection":
        return _gather_2d_reflect(img, x_idx, y_idx)
    elif padding_mode == "border":
        return _gather_2d_border(img, x_idx, y_idx)
    elif padding_mode == "zeros":
        return _gather_2d_zeros(img, x_idx, y_idx)
    else:
        raise ValueError(f"Unknown padding_mode: {padding_mode}")

def _compute_bilinear_indices_weights_2d(
    x_f: jnp.ndarray,
    y_f: jnp.ndarray
):
    """
    2D Bilinear interpolation indices & weights.
    """
    x0 = jnp.floor(x_f).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y_f).astype(jnp.int32)
    y1 = y0 + 1

    wx1 = x_f - x0
    wx0 = 1.0 - wx1
    wy1 = y_f - y0
    wy0 = 1.0 - wy1

    return (x0, x1, y0, y1, wx0, wx1, wy0, wy1)

def _compute_nearest_indices_2d(
    x_f: jnp.ndarray,
    y_f: jnp.ndarray
):
    """
    2D Nearest-neighbor:
    x_idx = round(x_f), y_idx = round(y_f)
    """
    x_idx = jnp.round(x_f).astype(jnp.int32)
    y_idx = jnp.round(y_f).astype(jnp.int32)
    return x_idx, y_idx

def _compute_bicubic_indices_weights_2d(
    x_f: jnp.ndarray,
    y_f: jnp.ndarray,
    cubic_A: float = -0.75
):
    """
    2D Bicubic interpolation indices & weights.
    4x4 neighborhood for each pixel in (x,y).
    """
    x_floor = jnp.floor(x_f).astype(jnp.int32)
    y_floor = jnp.floor(y_f).astype(jnp.int32)

    x_idx = jnp.stack([
        x_floor - 1,
        x_floor + 0,
        x_floor + 1,
        x_floor + 2,
    ], axis=-1)
    y_idx = jnp.stack([
        y_floor - 1,
        y_floor + 0,
        y_floor + 1,
        y_floor + 2,
    ], axis=-1)

    tx = jnp.expand_dims(x_f, axis=-1) - x_idx.astype(jnp.float32)
    ty = jnp.expand_dims(y_f, axis=-1) - y_idx.astype(jnp.float32)

    w_x = _cubic_kernel_weights(tx, A=cubic_A)
    w_y = _cubic_kernel_weights(ty, A=cubic_A)

    return x_idx, y_idx, w_x, w_y
