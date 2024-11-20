def __init__(self, model=None, device=None, img_size=(640, 640)):
    super(TracedModel, self).__init__()
    print(' Convert model to Traced-model... ')
    self.stride = model.stride
    self.names = model.names
    self.model = model
    self.model = revert_sync_batchnorm(self.model)
    self.model.to('cpu')
    self.model.eval()
    self.detect_layer = self.model.model[-1]
    self.model.traced = True
    rand_example = torch.rand(1, 3, img_size, img_size)
    traced_script_module = torch.jit.trace(self.model, rand_example, strict
        =False)
    traced_script_module.save('traced_model.pt')
    print(' traced_script_module saved! ')
    self.model = traced_script_module
    self.model.to(device)
    self.detect_layer.to(device)
    print(' model is traced! \n')
