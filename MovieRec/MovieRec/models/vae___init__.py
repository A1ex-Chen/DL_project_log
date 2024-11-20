def __init__(self, args):
    super().__init__(args)
    self.latent_dim = args.vae_latent_dim
    self.input_dropout = nn.Dropout(p=args.vae_dropout)
    dims = [args.vae_hidden_dim] * 2 * args.vae_num_hidden
    dims = [args.num_items] + dims + [args.vae_latent_dim * 2]
    encoder_modules, decoder_modules = [], []
    for i in range(len(dims) // 2):
        encoder_modules.append(nn.Linear(dims[2 * i], dims[2 * i + 1]))
        if i == 0:
            decoder_modules.append(nn.Linear(dims[-1] // 2, dims[-2]))
        else:
            decoder_modules.append(nn.Linear(dims[-2 * i - 1], dims[-2 * i -
                2]))
    self.encoder = nn.ModuleList(encoder_modules)
    self.decoder = nn.ModuleList(decoder_modules)
    self.encoder.apply(self.weight_init)
    self.decoder.apply(self.weight_init)
