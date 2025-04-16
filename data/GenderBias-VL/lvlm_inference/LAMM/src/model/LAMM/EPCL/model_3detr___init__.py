def __init__(self, args, input_dim):
    super().__init__()
    scale_ratio = args.preenc_npoints // args.vit_num_token
    self.layers = nn.ModuleList([PointnetSAModuleVotes(radius=0.2, nsample=
        64, npoint=args.preenc_npoints, mlp=[input_dim, 64, 256],
        normalize_xyz=True), PointnetSAModuleVotes(radius=0.2 * scale_ratio,
        nsample=64 * scale_ratio, npoint=args.vit_num_token, mlp=[256, 512,
        args.enc_dim], normalize_xyz=True)])
