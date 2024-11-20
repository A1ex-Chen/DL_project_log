def loading_data(data_dir):
    """ Use torchvision to load training, validation and test data
		Training data: andom scaling, cropping, flipping, resized data to 224x224
		Testing and validation data: resize and crop to the appropriate size
        returns datasets and data loaders
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/test'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(
        ), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406
        ], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128,
        shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=128)
    return (train_data, test_data, valid_data, trainloader, testloader,
        validloader)
