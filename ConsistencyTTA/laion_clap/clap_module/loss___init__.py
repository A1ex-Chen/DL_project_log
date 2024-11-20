def __init__(self, loss_name):
    super().__init__()
    if loss_name == 'bce':
        self.loss_func = nn.BCEWithLogitsLoss()
    elif loss_name == 'ce':
        self.loss_func = calc_celoss
    elif loss_name == 'mse':
        self.loss_func = nn.MSELoss()
    else:
        raise ValueError(
            f'the loss func should be at least one of [bce, ce, mse]')
