def replicate_first_beam(self, tensor, mask, beam_size: int):
    tensor = tensor.view(-1, beam_size, tensor.size(-1))
    tensor[mask] = tensor[mask][:, :1, :]
    return tensor.view(-1, tensor.size(-1))
