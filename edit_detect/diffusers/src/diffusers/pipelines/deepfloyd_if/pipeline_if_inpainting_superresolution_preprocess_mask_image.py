def preprocess_mask_image(self, mask_image) ->torch.Tensor:
    if not isinstance(mask_image, list):
        mask_image = [mask_image]
    if isinstance(mask_image[0], torch.Tensor):
        mask_image = torch.cat(mask_image, axis=0) if mask_image[0
            ].ndim == 4 else torch.stack(mask_image, axis=0)
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    elif isinstance(mask_image[0], PIL.Image.Image):
        new_mask_image = []
        for mask_image_ in mask_image:
            mask_image_ = mask_image_.convert('L')
            mask_image_ = resize(mask_image_, self.unet.config.sample_size)
            mask_image_ = np.array(mask_image_)
            mask_image_ = mask_image_[None, None, :]
            new_mask_image.append(mask_image_)
        mask_image = new_mask_image
        mask_image = np.concatenate(mask_image, axis=0)
        mask_image = mask_image.astype(np.float32) / 255.0
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
    elif isinstance(mask_image[0], np.ndarray):
        mask_image = np.concatenate([m[None, None, :] for m in mask_image],
            axis=0)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
    return mask_image
