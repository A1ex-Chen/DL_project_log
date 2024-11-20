def __init__(self, *, mlps: List[List[int]], radii: List[float], nsamples:
    List[int], post_mlp: List[int], bn: bool=True, use_xyz: bool=True,
    sample_uniformly: bool=False):
    super().__init__()
    assert len(mlps) == len(nsamples) == len(radii)
    self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn)
    self.groupers = nn.ModuleList()
    self.mlps = nn.ModuleList()
    for i in range(len(radii)):
        radius = radii[i]
        nsample = nsamples[i]
        self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample,
            use_xyz=use_xyz, sample_uniformly=sample_uniformly))
        mlp_spec = mlps[i]
        if use_xyz:
            mlp_spec[0] += 3
        self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))
