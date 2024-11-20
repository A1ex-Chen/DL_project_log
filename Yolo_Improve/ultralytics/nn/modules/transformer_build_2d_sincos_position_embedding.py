@staticmethod
def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0
    ):
    """Builds 2D sine-cosine position embedding."""
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / temperature ** omega
    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]
    return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
        torch.cos(out_h)], 1)[None]
