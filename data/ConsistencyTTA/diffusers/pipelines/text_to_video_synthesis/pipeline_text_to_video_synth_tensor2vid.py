def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ) ->List[np.ndarray]:
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(f, h, i * w, c)
    images = images.unbind(dim=0)
    images = [(image.cpu().numpy() * 255).astype('uint8') for image in images]
    return images
