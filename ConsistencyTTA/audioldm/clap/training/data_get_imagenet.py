def get_imagenet(args, preprocess_fns, split):
    assert split in ['train', 'val', 'v2']
    is_train = split == 'train'
    preprocess_train, preprocess_val = preprocess_fns
    if split == 'v2':
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=
            preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path
        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr
        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.
        batch_size, num_workers=args.workers, sampler=sampler)
    return DataInfo(dataloader, sampler)
