@staticmethod
def downsample(mask: torch.Tensor, batch_size: int, num_queries: int,
    value_embed_dim: int):
    """
        Downsamples the provided mask tensor to match the expected dimensions for scaled dot-product attention. If the
        aspect ratio of the mask does not match the aspect ratio of the output image, a warning is issued.

        Args:
            mask (`torch.Tensor`):
                The input mask tensor generated with `IPAdapterMaskProcessor.preprocess()`.
            batch_size (`int`):
                The batch size.
            num_queries (`int`):
                The number of queries.
            value_embed_dim (`int`):
                The dimensionality of the value embeddings.

        Returns:
            `torch.Tensor`:
                The downsampled mask tensor.

        """
    o_h = mask.shape[1]
    o_w = mask.shape[2]
    ratio = o_w / o_h
    mask_h = int(math.sqrt(num_queries / ratio))
    mask_h = int(mask_h) + int(num_queries % int(mask_h) != 0)
    mask_w = num_queries // mask_h
    mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w
        ), mode='bicubic').squeeze(0)
    if mask_downsample.shape[0] < batch_size:
        mask_downsample = mask_downsample.repeat(batch_size, 1, 1)
    mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)
    downsampled_area = mask_h * mask_w
    if downsampled_area < num_queries:
        warnings.warn(
            'The aspect ratio of the mask does not match the aspect ratio of the output image. Please update your masks or adjust the output size for optimal performance.'
            , UserWarning)
        mask_downsample = F.pad(mask_downsample, (0, num_queries -
            mask_downsample.shape[1]), value=0.0)
    if downsampled_area > num_queries:
        warnings.warn(
            'The aspect ratio of the mask does not match the aspect ratio of the output image. Please update your masks or adjust the output size for optimal performance.'
            , UserWarning)
        mask_downsample = mask_downsample[:, :num_queries]
    mask_downsample = mask_downsample.view(mask_downsample.shape[0],
        mask_downsample.shape[1], 1).repeat(1, 1, value_embed_dim)
    return mask_downsample
