def register_buffer(self, name, attr):
    if type(attr) == torch.Tensor:
        if attr.device != torch.device('cuda'):
            attr = attr.to(torch.device('cuda'))
    setattr(self, name, attr)
