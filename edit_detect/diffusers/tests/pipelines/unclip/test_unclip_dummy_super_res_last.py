@property
def dummy_super_res_last(self):
    torch.manual_seed(1)
    model = UNet2DModel(**self.dummy_super_res_kwargs)
    return model
