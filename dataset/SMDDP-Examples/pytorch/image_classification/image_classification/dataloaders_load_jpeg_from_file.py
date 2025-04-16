def load_jpeg_from_file(path, cuda=True):
    img_transforms = transforms.Compose([transforms.Resize(256), transforms
        .CenterCrop(224), transforms.ToTensor()])
    img = img_transforms(Image.open(path))
    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()
        input = img.unsqueeze(0).sub_(mean).div_(std)
    return input
