def build_encoder(args):
    if 'ViT-L' in args.clip_vit:
        args.enc_dim = 1024
    encoder = CLIPVITEncoder(vit_model=args.clip_vit, embed_dim=args.enc_dim)
    return encoder
