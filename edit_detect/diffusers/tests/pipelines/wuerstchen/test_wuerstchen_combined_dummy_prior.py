@property
def dummy_prior(self):
    torch.manual_seed(0)
    model_kwargs = {'c_in': 2, 'c': 8, 'depth': 2, 'c_cond': 32, 'c_r': 8,
        'nhead': 2}
    model = WuerstchenPrior(**model_kwargs)
    return model.eval()
