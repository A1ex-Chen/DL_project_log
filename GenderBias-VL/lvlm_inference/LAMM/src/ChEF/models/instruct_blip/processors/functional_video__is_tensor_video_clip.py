def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError('clip should be Tensor. Got %s' % type(clip))
    if not clip.ndimension() == 4:
        raise ValueError('clip should be 4D. Got %dD' % clip.dim())
    return True
