def create_from(self, module):
    assert isinstance(module, torch.nn.Module)
    if issubclass(self.replaceCls, GenericMixin):
        new_class = type('{}MixedWith{}'.format(self.replaceCls.__name__,
            module.__class__.__name__), (self.replaceCls, module.__class__), {}
            )
        module.__class__ = new_class
    else:
        module.__class__ = self.replaceCls
    if isinstance(module, Caffe2Compatible):
        module.tensor_mode = False
    return module
