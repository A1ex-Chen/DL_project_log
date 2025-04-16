def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=
    32, embed_dim=64, n_embed=512, decay=0.99):
    super().__init__()
    self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel,
        stride=4)
    self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2
        )
    self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
    self.quantize_t = Quantize(embed_dim, n_embed)
    self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block,
        n_res_channel, stride=2)
    self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
    self.quantize_b = Quantize(embed_dim, n_embed)
    self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2,
        padding=1)
    self.dec = Decoder(embed_dim + embed_dim, in_channel, channel,
        n_res_block, n_res_channel, stride=4)
