@property
def dummy_super_res_first(self):
    torch.manual_seed(0)
    model = UNet2DModel(**self.dummy_super_res_kwargs)
    return model
