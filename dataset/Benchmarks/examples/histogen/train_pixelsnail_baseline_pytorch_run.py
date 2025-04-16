def run(params):
    args = candle.ArgumentStruct(**params)
    ndevices = torch.cuda.device_count()
    if ndevices < 1:
        raise Exception('No CUDA gpus available')
    device = 'cuda'
    dataset = LMDBDataset(args.lmdb_filename)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True)
    ckpt = {}
    if args.ckpt_restart is not None:
        ckpt = torch.load(args.ckpt_restart)
        args = ckpt['args']
    if args.hier == 'top':
        model = PixelSNAIL([32, 32], 512, args.channel, 5, 4, args.
            n_res_block, args.n_res_channel, dropout=args.dropout,
            n_out_res_block=args.n_out_res_block)
    elif args.hier == 'bottom':
        model = PixelSNAIL([64, 64], 512, args.channel, 5, 4, args.
            n_res_block, args.n_res_channel, attention=False, dropout=args.
            dropout, n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)
    model = nn.DataParallel(model)
    model = model.to(device)
    scheduler = None
    if args.sched_mode == 'cycle':
        scheduler = CycleScheduler(optimizer, args.learning_rate, n_iter=
            len(loader) * args.epochs, momentum=None)
    for i in range(args.epochs):
        train(args, i, loader, model, optimizer, scheduler, device)
        torch.save({'model': model.module.state_dict(), 'args': args},
            f'{args.ckpt_directory}/checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt'
            )
