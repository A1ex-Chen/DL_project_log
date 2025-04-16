@staticmethod
def from_tensors(tensors: List[torch.Tensor], size_divisibility: int=0,
    pad_value: float=0.0) ->'ImageList':
    """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[:-2] == tensors[0].shape[:-2], t.shape
    image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
    image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values
    if size_divisibility > 1:
        stride = size_divisibility
        max_size = (max_size + (stride - 1)).div(stride, rounding_mode='floor'
            ) * stride
    if torch.jit.is_scripting():
        max_size: List[int] = max_size.to(dtype=torch.long).tolist()
    elif torch.jit.is_tracing():
        image_sizes = image_sizes_tensor
    if len(tensors) == 1:
        image_size = image_sizes[0]
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] -
            image_size[0]]
        batched_imgs = F.pad(tensors[0], padding_size, value=pad_value
            ).unsqueeze_(0)
    else:
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(
            max_size)
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return ImageList(batched_imgs.contiguous(), image_sizes)
