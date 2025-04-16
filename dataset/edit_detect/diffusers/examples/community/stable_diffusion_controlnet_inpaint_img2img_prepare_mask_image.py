def prepare_mask_image(mask_image):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]
        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.
            Image.Image):
            mask_image = np.concatenate([np.array(m.convert('L'))[None,
                None, :] for m in mask_image], axis=0)
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.
            ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in
                mask_image], axis=0)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
    return mask_image
