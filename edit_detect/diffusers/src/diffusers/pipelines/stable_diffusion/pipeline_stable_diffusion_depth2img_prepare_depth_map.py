def prepare_depth_map(self, image, depth_map, batch_size,
    do_classifier_free_guidance, dtype, device):
    if isinstance(image, PIL.Image.Image):
        image = [image]
    else:
        image = list(image)
    if isinstance(image[0], PIL.Image.Image):
        width, height = image[0].size
    elif isinstance(image[0], np.ndarray):
        width, height = image[0].shape[:-1]
    else:
        height, width = image[0].shape[-2:]
    if depth_map is None:
        pixel_values = self.feature_extractor(images=image, return_tensors='pt'
            ).pixel_values
        pixel_values = pixel_values.to(device=device)
        if torch.backends.mps.is_available():
            autocast_ctx = contextlib.nullcontext()
            logger.warning(
                'The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16, but autocast is not yet supported on MPS.'
                )
        else:
            autocast_ctx = torch.autocast(device.type, dtype=dtype)
        with autocast_ctx:
            depth_map = self.depth_estimator(pixel_values).predicted_depth
    else:
        depth_map = depth_map.to(device=device, dtype=dtype)
    depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1),
        size=(height // self.vae_scale_factor, width // self.
        vae_scale_factor), mode='bicubic', align_corners=False)
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = depth_map.to(dtype)
    if depth_map.shape[0] < batch_size:
        repeat_by = batch_size // depth_map.shape[0]
        depth_map = depth_map.repeat(repeat_by, 1, 1, 1)
    depth_map = torch.cat([depth_map] * 2
        ) if do_classifier_free_guidance else depth_map
    return depth_map
