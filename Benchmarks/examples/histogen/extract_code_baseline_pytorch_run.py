def run(params):
    args = candle.ArgumentStruct(**params)
    ndevices = torch.cuda.device_count()
    if ndevices < 1:
        raise Exception('No CUDA gpus available')
    device = 'cuda'
    transform = transforms.Compose([transforms.Resize(args.size),
        transforms.CenterCrop(args.size), transforms.ToTensor(), transforms
        .Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageFileDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4)
    model = VQVAE()
    if args.ckpt_restart is not None:
        model.load_state_dict(torch.load(args.ckpt_restart))
    model = model.to(device)
    model.eval()
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(args.lmdb_filename, map_size=map_size)
    extract(env, loader, model, device)
