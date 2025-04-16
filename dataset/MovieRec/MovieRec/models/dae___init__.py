def __init__(self, args):
    super().__init__(args)
    self.input_dropout = nn.Dropout(p=args.dae_dropout)
    dims = [args.dae_hidden_dim] * 2 * args.dae_num_hidden
    dims = [args.num_items] + dims + [args.dae_latent_dim]
    encoder_modules, decoder_modules = [], []
    for i in range(len(dims) // 2):
        encoder_modules.append(nn.Linear(dims[2 * i], dims[2 * i + 1]))
        decoder_modules.append(nn.Linear(dims[-2 * i - 1], dims[-2 * i - 2]))
    self.encoder = nn.ModuleList(encoder_modules)
    self.decoder = nn.ModuleList(decoder_modules)
    self.encoder.apply(self.weight_init)
    self.decoder.apply(self.weight_init)
