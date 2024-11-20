def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]
    ) ->torch.Tensor:
    if self.remap is not None:
        indices = indices.reshape(shape[0], -1)
        indices = self.unmap_to_all(indices)
        indices = indices.reshape(-1)
    z_q: torch.Tensor = self.embedding(indices)
    if shape is not None:
        z_q = z_q.view(shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
    return z_q
