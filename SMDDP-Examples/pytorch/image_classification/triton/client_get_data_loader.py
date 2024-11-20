def get_data_loader(batch_size, *, data_path):
    valdir = os.path.join(data_path, 'val-jpeg')
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.
        ToTensor()]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=
        batch_size, shuffle=False)
    return val_loader
