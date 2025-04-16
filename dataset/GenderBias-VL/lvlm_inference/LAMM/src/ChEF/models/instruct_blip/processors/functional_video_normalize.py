def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError('clip should be a 4D torch.tensor')
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip
