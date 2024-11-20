def mask_pil_to_torch(mask, height, width):
    if isinstance(mask, Union[PIL.Image.Image, np.ndarray]):
        mask = [mask]
    if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
        mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in
            mask]
        mask = np.concatenate([np.array(m.convert('L'))[None, None, :] for
            m in mask], axis=0)
        mask = mask.astype(np.float32) / 255.0
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)
    mask = torch.from_numpy(mask)
    return mask
