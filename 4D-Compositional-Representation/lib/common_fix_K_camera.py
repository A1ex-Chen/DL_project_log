def fix_K_camera(K, img_size=137):
    """Fix camera projection matrix.
    This changes a camera projection matrix that maps to
    [0, img_size] x [0, img_size] to one that maps to [-1, 1] x [-1, 1].

    Args:
        K (np.ndarray):     Camera projection matrix.
        img_size (float):   Size of image plane K projects to.
    """
    scale_mat = torch.tensor([[2.0 / img_size, 0, -1], [0, 2.0 / img_size, 
        -1], [0, 0, 1.0]], device=K.device, dtype=K.dtype)
    K_new = scale_mat.view(1, 3, 3) @ K
    return K_new
