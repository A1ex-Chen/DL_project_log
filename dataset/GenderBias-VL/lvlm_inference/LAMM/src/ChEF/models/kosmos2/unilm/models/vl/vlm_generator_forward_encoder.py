@torch.jit.export
def forward_encoder(self, net_input: Dict[str, Tensor]):
    if not self.has_encoder():
        return None
    return [model.encoder.forward_torchscript(net_input) for model in self.
        models]
