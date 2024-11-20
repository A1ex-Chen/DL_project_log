@staticmethod
def _init_weight(out: nn.Parameter):
    """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
    n_pos, dim = out.shape
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for
        j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    sentinel = dim // 2 if dim % 2 == 0 else dim // 2 + 1
    out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    return out
