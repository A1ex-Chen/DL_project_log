def get_model_optimizer(self, resolution=32):
    set_seed(0)
    model = UNet2DModel(sample_size=resolution, in_channels=3, out_channels=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    return model, optimizer
