def _preprocess_mask(mask: Union[List, PIL.Image.Image, torch.Tensor]):
    if isinstance(mask, torch.Tensor):
        return mask
    elif isinstance(mask, PIL.Image.Image):
        mask = [mask]
    if isinstance(mask[0], PIL.Image.Image):
        w, h = mask[0].size
        w, h = (x - x % 32 for x in (w, h))
        mask = [np.array(m.convert('L').resize((w, h), resample=
            PIL_INTERPOLATION['nearest']))[None, :] for m in mask]
        mask = np.concatenate(mask, axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
    elif isinstance(mask[0], torch.Tensor):
        mask = torch.cat(mask, dim=0)
    return mask
