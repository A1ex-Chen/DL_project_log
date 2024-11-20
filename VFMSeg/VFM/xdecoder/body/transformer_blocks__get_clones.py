def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
