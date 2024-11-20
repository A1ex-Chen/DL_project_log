@staticmethod
def get_data_loader(args, cfg, data_dict):
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])
    class_names = data_dict['names']
    assert len(class_names
        ) == nc, f'the length of class names does not match the number of classes defined'
    grid_size = max(int(max(cfg.model.head.strides)), 32)
    train_loader = create_dataloader(train_path, args.img_size, args.
        batch_size // args.world_size, grid_size, hyp=dict(cfg.data_aug),
        augment=True, rect=args.rect, rank=args.local_rank, workers=args.
        workers, shuffle=True, check_images=args.check_images, check_labels
        =args.check_labels, data_dict=data_dict, task='train',
        specific_shape=args.specific_shape, height=args.height, width=args.
        width, cache_ram=args.cache_ram)[0]
    val_loader = None
    if args.rank in [-1, 0]:
        val_loader = create_dataloader(val_path, args.img_size, args.
            batch_size // args.world_size * 2, grid_size, hyp=dict(cfg.
            data_aug), rect=True, rank=-1, pad=0.5, workers=args.workers,
            check_images=args.check_images, check_labels=args.check_labels,
            data_dict=data_dict, task='val', specific_shape=args.
            specific_shape, height=args.height, width=args.width, cache_ram
            =args.cache_ram)[0]
    return train_loader, val_loader
