def build_epcl_encoder(pretrain=True, store_path='../model_zoo/EPCL_ckpts/',
    num_token=256, device='cpu'):
    if pretrain and not os.path.exists(store_path):
        raise ValueError(f'EPCL pretrained model not found at [{store_path}]!')
    ckpt = torch.load(store_path, map_location=device)
    args = ckpt['args']
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    model = Model3DETR_encoder(pre_encoder, encoder, dataset_name=args.
        dataset_name, encoder_dim=args.enc_dim, decoder_dim=args.dec_dim,
        num_queries=args.nqueries, use_task_emb=args.use_task_emb, vit_only
        =True)
    print('===> Loading EPCL encoder...\n')
    match_keys = []
    for key in model.state_dict().keys():
        if key in ckpt['model']:
            match_keys.append(key)
    model.load_state_dict(ckpt['model'], strict=False)
    print(
        f"""===> Loaded
	 {store_path}; {len(match_keys)} keys matched; {len(model.state_dict().keys())} keys in total."""
        )
    return model.to(device)
