def main():
    args = parse_args()
    ckpt_dir = os.path.join(args.results_dir, f'{args.encoder}_NBt1D')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    model, device = build_model(args)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.
        momentum, weight_decay=args.weight_decay)
    if args.weight_file:
        if device.type == 'cuda':
            checkpoint = torch.load(args.weight_file)
        else:
            checkpoint = torch.load(args.weight_file, map_location=lambda
                storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print("=> loaded checkpoint '{}' (epoch {})".format(args.
            weight_file, checkpoint['epoch']))
    else:
        start_epoch = 0
    train_batches, validation_batches, dataset_info = get_data(args)
    n_train_images = dataset_info.splits['train'].num_examples
    n_val_images = dataset_info.splits['validation'].num_examples
    log_keys = ['acc_train_top-1', 'acc_train_top-5', 'acc_val_top-1',
        'acc_val_top-5']
    log_keys_for_csv = log_keys.copy()
    log_keys_for_csv.extend(['loss_train', 'loss_val', 'epoch', 'lr'])
    csvlogger = CSVLogger(log_keys_for_csv, os.path.join(ckpt_dir,
        'logs.csv'), append=True)
    best_acc1 = -1
    for epoch in range(start_epoch, args.epochs):
        if not args.finetune:
            lr = adjust_learning_rate(optimizer, epoch, args)
        else:
            lr = args.lr
        logs = train(train_batches, model, criterion, optimizer, epoch,
            device, n_train_images, args)
        logs = validate(validation_batches, model, criterion, device,
            n_val_images, logs, args)
        is_best = logs['acc_val_top-1'] > best_acc1
        best_acc1 = max(logs['acc_val_top-1'], best_acc1)
        save_ckpt(ckpt_dir, model, optimizer, epoch, is_best)
        logs['epoch'] = epoch
        logs['lr'] = lr
        logs.pop('time', None)
        csvlogger.write_logs(logs)
    print('done')
