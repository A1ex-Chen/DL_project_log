def preprocess_mask(mask, scale_factor=8):
    mask = mask.convert('L')
    w, h = mask.size
    w, h = (x - x % 32 for x in (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=
        PIL_INTERPOLATION['nearest'])
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = 1 - mask
    return mask
