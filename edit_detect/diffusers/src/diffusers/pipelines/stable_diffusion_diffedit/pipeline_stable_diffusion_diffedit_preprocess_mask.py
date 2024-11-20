def preprocess_mask(mask, batch_size: int=1):
    if not isinstance(mask, torch.Tensor):
        if isinstance(mask, PIL.Image.Image) or isinstance(mask, np.ndarray):
            mask = [mask]
        if isinstance(mask, list):
            if isinstance(mask[0], PIL.Image.Image):
                mask = [(np.array(m.convert('L')).astype(np.float32) / 
                    255.0) for m in mask]
            if isinstance(mask[0], np.ndarray):
                mask = np.stack(mask, axis=0) if mask[0
                    ].ndim < 3 else np.concatenate(mask, axis=0)
                mask = torch.from_numpy(mask)
            elif isinstance(mask[0], torch.Tensor):
                mask = torch.stack(mask, dim=0) if mask[0
                    ].ndim < 3 else torch.cat(mask, dim=0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
    if batch_size > 1:
        if mask.shape[0] == 1:
            mask = torch.cat([mask] * batch_size)
        elif mask.shape[0] > 1 and mask.shape[0] != batch_size:
            raise ValueError(
                f'`mask_image` with batch size {mask.shape[0]} cannot be broadcasted to batch size {batch_size} inferred by prompt inputs'
                )
    if mask.shape[1] != 1:
        raise ValueError(
            f'`mask_image` must have 1 channel, but has {mask.shape[1]} channels'
            )
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError('`mask_image` should be in [0, 1] range')
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask
