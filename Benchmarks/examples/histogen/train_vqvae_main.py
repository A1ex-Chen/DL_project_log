def main(args):
    device = 'cuda'
    args.distributed = dist.get_world_size() > 1
    transform = transforms.Compose([transforms.Resize(args.size),
        transforms.CenterCrop(args.size), transforms.ToTensor(), transforms
        .Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.
        distributed)
    loader = DataLoader(dataset, batch_size=128 // args.n_gpu, sampler=
        sampler, num_workers=2)
    model = VQVAE().to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist
            .get_local_rank()], output_device=dist.get_local_rank())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(loader) *
            args.epoch, momentum=None, warmup_proportion=0.05)
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        if dist.is_primary():
            torch.save(model.state_dict(),
                f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt')
