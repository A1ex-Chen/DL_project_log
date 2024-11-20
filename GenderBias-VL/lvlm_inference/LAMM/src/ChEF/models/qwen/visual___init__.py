def __init__(self, image_size: int, patch_size: int, width: int, layers:
    int, heads: int, mlp_ratio: float, n_queries: int=256, output_dim: int=
    512, **kwargs):
    super().__init__()
    image_height, image_width = self.image_size = image_size, image_size
    patch_height, patch_width = self.patch_size = patch_size, patch_size
    self.grid_size = image_height // patch_height, image_width // patch_width
    self.output_dim = output_dim
    mean = 0.48145466, 0.4578275, 0.40821073
    std = 0.26862954, 0.26130258, 0.27577711
    self.image_transform = transforms.Compose([transforms.Resize((
        image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=
        patch_size, stride=patch_size, bias=False)
    scale = width ** -0.5
    self.positional_embedding = nn.Parameter(scale * torch.randn(256, width))
    norm_layer = partial(nn.LayerNorm, eps=1e-06)
    act_layer = nn.GELU
    self.ln_pre = norm_layer(width)
    self.transformer = TransformerBlock(width, layers, heads, mlp_ratio,
        act_layer=act_layer, norm_layer=norm_layer)
    self.attn_pool = Resampler(grid_size=int(math.sqrt(n_queries)),
        embed_dim=output_dim, num_heads=output_dim // 128, kv_dim=width,
        norm_layer=norm_layer)
    self.ln_post = norm_layer(output_dim)
    self.proj = nn.Parameter(output_dim ** -0.5 * torch.randn(output_dim,
        output_dim))
