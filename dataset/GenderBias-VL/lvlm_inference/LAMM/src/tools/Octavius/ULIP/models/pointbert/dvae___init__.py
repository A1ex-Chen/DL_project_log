def __init__(self, config, **kwargs):
    super().__init__()
    self.group_size = config.group_size
    self.num_group = config.num_group
    self.encoder_dims = config.encoder_dims
    self.tokens_dims = config.tokens_dims
    self.decoder_dims = config.decoder_dims
    self.num_tokens = config.num_tokens
    self.group_divider = Group(num_group=self.num_group, group_size=self.
        group_size)
    self.encoder = Encoder(encoder_channel=self.encoder_dims)
    self.dgcnn_1 = DGCNN(encoder_channel=self.encoder_dims, output_channel=
        self.num_tokens)
    self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims)
        )
    self.dgcnn_2 = DGCNN(encoder_channel=self.tokens_dims, output_channel=
        self.decoder_dims)
    self.decoder = Decoder(encoder_channel=self.decoder_dims, num_fine=self
        .group_size)
