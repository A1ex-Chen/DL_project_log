def __init__(self, params: ModelArgs):
    """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers
    self.tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim,
        init_method=lambda x: x)
    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
        self.layers.append(TransformerBlock(layer_id, params))
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=
        False, init_method=lambda x: x)
    self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.
        n_heads, self.params.max_seq_len * 2)
