def __init__(self, input_resolution: int, patch_size: int, width: int,
    layers: int, heads: int, use_grad_checkpointing: bool):
    super().__init__()
    self.input_resolution = input_resolution
    self.num_features = width
    self.num_heads = heads
    self.num_patches = (input_resolution // patch_size) ** 2
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=
        patch_size, stride=patch_size, bias=False)
    scale = width ** -0.5
    self.class_embedding = nn.Parameter(scale * torch.randn(width))
    self.positional_embedding = nn.Parameter(scale * torch.randn(self.
        num_patches + 1, width))
    self.ln_pre = LayerNorm(width)
    self.transformer = Transformer(width, layers, heads,
        use_grad_checkpointing=use_grad_checkpointing)
