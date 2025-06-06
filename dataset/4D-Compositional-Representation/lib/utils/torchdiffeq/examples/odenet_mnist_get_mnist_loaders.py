def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000,
    perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([transforms.RandomCrop(28,
            padding=4), transforms.ToTensor()])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST(root='.data/mnist', train=True,
        download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True)
    train_eval_loader = DataLoader(datasets.MNIST(root='.data/mnist', train
        =True, download=True, transform=transform_test), batch_size=
        test_batch_size, shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(datasets.MNIST(root='.data/mnist', train=False,
        download=True, transform=transform_test), batch_size=
        test_batch_size, shuffle=False, num_workers=2, drop_last=True)
    return train_loader, test_loader, train_eval_loader
