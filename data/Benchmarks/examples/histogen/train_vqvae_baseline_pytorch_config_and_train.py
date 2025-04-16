def config_and_train(args):
    ndevices = torch.cuda.device_count()
    if ndevices < 1:
        raise Exception('No CUDA gpus available')
    device = 'cuda'
    args.distributed = dist.get_world_size() > 1
    transform = transforms.Compose([transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.
        distributed)
    loader = DataLoader(dataset, batch_size=args.batch_size // args.
        n_gpu_per_machine, sampler=sampler, num_workers=2)
    model = VQVAE().to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist
            .get_local_rank()], output_device=dist.get_local_rank())
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.sched_mode == 'cycle':
        scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(loader) *
            args.epochs, momentum=None, warmup_proportion=0.05)
    for i in range(args.epochs):
        train(i, loader, model, optimizer, scheduler, device)
        if dist.is_primary():
            torch.save(model.state_dict(),
                f'{args.ckpt_directory}/checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
                )
