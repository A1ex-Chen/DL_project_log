def __init__(self, input_dim, output_dim, args):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
    self.latent_query = torch.nn.Parameter(torch.randn(args.
        latent_query_num, output_dim))
    self.x_attn = MultiheadAttention(output_dim, args.
        decoder_attention_heads, kdim=output_dim, vdim=output_dim, dropout=
        args.attention_dropout, encoder_decoder_attention=True)
