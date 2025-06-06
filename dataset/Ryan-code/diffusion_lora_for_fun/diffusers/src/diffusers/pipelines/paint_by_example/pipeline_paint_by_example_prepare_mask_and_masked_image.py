def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Paint by Example pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                f'`image` is a torch.Tensor but `mask` (type: {type(mask)} is not'
                )
        if image.ndim == 3:
            assert image.shape[0
                ] == 3, 'Image outside a batch should be of shape (3, H, W)'
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.ndim == 3:
            if mask.shape[0] == image.shape[0]:
                mask = mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(0)
        assert image.ndim == 4 and mask.ndim == 4, 'Image and Mask must have 4 dimensions'
        assert image.shape[-2:] == mask.shape[-2:
            ], 'Image and Mask must have the same spatial dimensions'
        assert image.shape[0] == mask.shape[0
            ], 'Image and Mask must have the same batch size'
        assert mask.shape[1] == 1, 'Mask image must have a single channel'
        if image.min() < -1 or image.max() > 1:
            raise ValueError('Image should be in [-1, 1] range')
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError('Mask should be in [0, 1] range')
        mask = 1 - mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(
            f'`mask` is a torch.Tensor but `image` (type: {type(image)} is not'
            )
    else:
        if isinstance(image, PIL.Image.Image):
            image = [image]
        image = np.concatenate([np.array(i.convert('RGB'))[None, :] for i in
            image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        if isinstance(mask, PIL.Image.Image):
            mask = [mask]
        mask = np.concatenate([np.array(m.convert('L'))[None, None, :] for
            m in mask], axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask = 1 - mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
    masked_image = image * mask
    return mask, masked_image
