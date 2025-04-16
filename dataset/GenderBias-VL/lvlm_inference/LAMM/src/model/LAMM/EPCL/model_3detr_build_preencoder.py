def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    if args.pointnet_downsample:
        preencoder = Pointnet2Preencoder(args, 3 * int(args.use_color))
    else:
        preencoder = PointnetSAModuleVotes(radius=args.preenc_radius,
            nsample=64, npoint=args.preenc_npoints, mlp=mlp_dims,
            normalize_xyz=True)
    return preencoder
