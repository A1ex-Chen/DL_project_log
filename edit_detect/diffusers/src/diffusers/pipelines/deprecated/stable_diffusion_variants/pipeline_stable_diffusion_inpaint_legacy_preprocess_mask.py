def preprocess_mask(mask, batch_size, scale_factor=8):
    if not isinstance(mask, torch.Tensor):
        mask = mask.convert('L')
        w, h = mask.size
        w, h = (x - x % 8 for x in (w, h))
        mask = mask.resize((w // scale_factor, h // scale_factor), resample
            =PIL_INTERPOLATION['nearest'])
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = np.vstack([mask[None]] * batch_size)
        mask = 1 - mask
        mask = torch.from_numpy(mask)
        return mask
    else:
        valid_mask_channel_sizes = [1, 3]
        if mask.shape[3] in valid_mask_channel_sizes:
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] not in valid_mask_channel_sizes:
            raise ValueError(
                f'Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension, but received mask of shape {tuple(mask.shape)}'
                )
        mask = mask.mean(dim=1, keepdim=True)
        h, w = mask.shape[-2:]
        h, w = (x - x % 8 for x in (h, w))
        mask = torch.nn.functional.interpolate(mask, (h // scale_factor, w //
            scale_factor))
        return mask
