def concat_elu(x):
    """like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU"""
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))
