def gelu(self, gate):
    if gate.device.type != 'mps':
        return F.gelu(gate)
    return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)
