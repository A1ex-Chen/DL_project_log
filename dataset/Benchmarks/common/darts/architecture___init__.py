def __init__(self, model, args, hyperparams=Hyperparameters(), device='cpu'):
    self.momentum = args.momentum
    self.wd = args.weight_decay
    self.model = model
    self.device = device
    self.optimizer = optim.Adam(self.model.arch_parameters(), lr=
        hyperparams.alpha_lr, betas=(0.5, 0.999), weight_decay=hyperparams.
        alpha_wd)
