def __init__(self, args, **kwargs):
    super().__init__()
    self.args = args
    self.beit3 = BEiT3(args)
    self.num_img_patches = self.beit3.vision_embed.num_position_embeddings()
    self.hidden_size = args.encoder_embed_dim
