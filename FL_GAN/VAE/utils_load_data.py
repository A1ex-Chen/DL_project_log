def load_data(dataset):
    """Load (training and test set)."""
    data_transform = [transforms.ToTensor()]
    if dataset == 'mnist':
        data_transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        data_transform = transforms.Compose(data_transform)
        my_train = MNIST(root='./data', train=True, transform=
            data_transform, download=True)
        my_test = MNIST(root='./data', train=False, transform=
            data_transform, download=True)
    elif dataset == 'fashion-mnist':
        data_transform.append(transforms.Normalize((0.5,), (0.5,)))
        data_transform = transforms.Compose(data_transform)
        my_train = FashionMNIST(root='./data', train=True, transform=
            data_transform, download=True)
        my_test = FashionMNIST(root='./data', train=False, transform=
            data_transform, download=True)
    elif dataset == 'cifar':
        data_transform.append(transforms.Normalize(mean=[0.485, 0.456, 
            0.406], std=[0.229, 0.224, 0.225]))
        data_transform = transforms.Compose(data_transform)
        my_train = CIFAR10(root='./data', train=True, transform=
            data_transform, download=True)
        my_test = CIFAR10(root='./data', train=False, transform=
            data_transform, download=True)
    elif dataset == 'stl':
        data_transform.append(transforms.Normalize(mean=[0.4914, 0.4822, 
            0.4465], std=[0.2471, 0.2435, 0.2616]))
        data_transform = transforms.Compose(data_transform)
        my_train = STL10(root='./data', split='unlabeled', transform=
            data_transform, download=True)
        my_test = STL10(root='./data', split='test', transform=
            data_transform, download=True)
    return my_train, my_test
