def __init__(self, kernel_size: Tuple[int, int]=(16, 16), stride: Tuple[int,
    int]=(16, 16), padding: Tuple[int, int]=(0, 0), in_chans: int=3,
    embed_dim: int=768) ->None:
    """
        Initialize PatchEmbed module.

        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
    super().__init__()
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
        stride=stride, padding=padding)
