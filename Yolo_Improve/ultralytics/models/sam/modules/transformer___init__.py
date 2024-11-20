def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int=1
    ) ->None:
    """
        Initializes the Attention model with the given dimensions and settings.

        Args:
            embedding_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            downsample_rate (int, optional): The factor by which the internal dimensions are downsampled. Defaults to 1.

        Raises:
            AssertionError: If 'num_heads' does not evenly divide the internal dim (embedding_dim / downsample_rate).
        """
    super().__init__()
    self.embedding_dim = embedding_dim
    self.internal_dim = embedding_dim // downsample_rate
    self.num_heads = num_heads
    assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'
    self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
    self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
    self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
    self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
