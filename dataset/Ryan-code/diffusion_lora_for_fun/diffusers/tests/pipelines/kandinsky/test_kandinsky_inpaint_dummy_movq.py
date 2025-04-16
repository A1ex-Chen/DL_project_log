@property
def dummy_movq(self):
    torch.manual_seed(0)
    model = VQModel(**self.dummy_movq_kwargs)
    return model
