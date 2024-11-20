def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (C, T, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError('clip should be a 4D torch.tensor')
    return clip.flip(-1)
