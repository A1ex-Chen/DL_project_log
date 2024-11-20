def load_model(model, checkpoint, device):
    ndevices = torch.cuda.device_count()
    if ndevices == 0:
        ckpt = torch.load(os.path.join('checkpoint', checkpoint),
            map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(os.path.join('checkpoint', checkpoint))
    if 'args' in ckpt:
        args = ckpt['args']
    if model == 'vqvae':
        model = VQVAE()
    elif model == 'pixelsnail_top':
        model = PixelSNAIL([32, 32], 512, args.channel, 5, 4, args.
            n_res_block, args.n_res_channel, dropout=args.dropout,
            n_out_res_block=args.n_out_res_block)
    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL([64, 64], 512, args.channel, 5, 4, args.
            n_res_block, args.n_res_channel, attention=False, dropout=args.
            dropout, n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model
