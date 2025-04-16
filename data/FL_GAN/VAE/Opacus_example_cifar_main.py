def main():
    args = parse_args()
    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)
    if args.local_rank != -1 or args.device != 'cpu':
        rank, local_rank, world_size = setup(args)
        device = local_rank
    else:
        device = 'cpu'
        rank = 0
        world_size = 1
    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = (
                'To use secure RNG, you must install the torchcsprng package! Check out the instructions here: https://github.com/pytorch/csprng#installation'
                )
            raise ImportError(msg) from e
        generator = prng.create_random_device_generator('/dev/urandom')
    else:
        generator = None
    augmentations = [transforms.RandomCrop(32, padding=4), transforms.
        RandomHorizontalFlip()]
    normalize = [transforms.ToTensor(), transforms.Normalize((0.4914, 
        0.4822, 0.4465), (0.2023, 0.1994, 0.201))]
    train_transform = transforms.Compose(augmentations + normalize if args.
        disable_dp else normalize)
    test_transform = transforms.Compose(normalize)
    train_dataset = CIFAR10(root=args.data_root, train=True, download=True,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=
        args.batch_size, generator=generator, num_workers=args.workers,
        pin_memory=True)
    test_dataset = CIFAR10(root=args.data_root, train=False, download=True,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args
        .batch_size_test, shuffle=False, num_workers=args.workers)
    best_acc1 = 0
    model = convnet(num_classes=10)
    model = model.to(device)
    if world_size > 1:
        if not args.disable_dp:
            if args.clip_per_layer:
                model = DDP(model, device_ids=[device])
            else:
                model = DPDDP(model)
        else:
            model = DDP(model, device_ids=[device])
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args
            .momentum, weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(
            'Optimizer not recognized. Please check spelling')
    privacy_engine = None
    if not args.disable_dp:
        if args.clip_per_layer:
            n_layers = len([(n, p) for n, p in model.named_parameters() if
                p.requires_grad])
            max_grad_norm = [args.max_per_sample_grad_norm / np.sqrt(n_layers)
                ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        clipping = 'per_layer' if args.clip_per_layer else 'flat'
        model, optimizer, train_loader = privacy_engine.make_private(module
            =model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=args.sigma, max_grad_norm=max_grad_norm,
            clipping=clipping, grad_sample_mode=args.grad_sample_mode)
    accuracy_per_epoch = []
    time_per_epoch = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.lr_schedule == 'cos':
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1))
                )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train_duration = train(args, model, train_loader, optimizer,
            privacy_engine, epoch, device)
        top1_acc = test(args, model, test_loader, device)
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)
        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))
        save_checkpoint({'epoch': epoch + 1, 'arch': 'Convnet',
            'state_dict': model.state_dict(), 'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict()}, is_best, filename=args.
            checkpoint_file + '.tar')
    if rank == 0:
        time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
        avg_time_per_epoch = sum(time_per_epoch_seconds) / len(
            time_per_epoch_seconds)
        metrics = {'accuracy': best_acc1, 'accuracy_per_epoch':
            accuracy_per_epoch, 'avg_time_per_epoch_str': str(timedelta(
            seconds=int(avg_time_per_epoch))), 'time_per_epoch':
            time_per_epoch_seconds}
        logger.info(
            """
Note:
- 'total_time' includes the data loading time, training time and testing time.
- 'time_per_epoch' measures the training time only.
"""
            )
        logger.info(metrics)
    if world_size > 1:
        cleanup()
