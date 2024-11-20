def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError('clip should be a 4D torch.tensor')
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError('height and width must be no smaller than crop_size')
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)
