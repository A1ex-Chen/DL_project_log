def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError('clip should be a 4D tensor')
    return clip[..., i:i + h, j:j + w]
