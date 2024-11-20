def prepare_controlnet_conditioning_image(controlnet_conditioning_image,
    width, height, batch_size, num_images_per_prompt, device, dtype,
    do_classifier_free_guidance):
    if not isinstance(controlnet_conditioning_image, torch.Tensor):
        if isinstance(controlnet_conditioning_image, PIL.Image.Image):
            controlnet_conditioning_image = [controlnet_conditioning_image]
        if isinstance(controlnet_conditioning_image[0], PIL.Image.Image):
            controlnet_conditioning_image = [np.array(i.resize((width,
                height), resample=PIL_INTERPOLATION['lanczos']))[None, :] for
                i in controlnet_conditioning_image]
            controlnet_conditioning_image = np.concatenate(
                controlnet_conditioning_image, axis=0)
            controlnet_conditioning_image = np.array(
                controlnet_conditioning_image).astype(np.float32) / 255.0
            controlnet_conditioning_image = (controlnet_conditioning_image.
                transpose(0, 3, 1, 2))
            controlnet_conditioning_image = torch.from_numpy(
                controlnet_conditioning_image)
        elif isinstance(controlnet_conditioning_image[0], torch.Tensor):
            controlnet_conditioning_image = torch.cat(
                controlnet_conditioning_image, dim=0)
    image_batch_size = controlnet_conditioning_image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        repeat_by = num_images_per_prompt
    controlnet_conditioning_image = (controlnet_conditioning_image.
        repeat_interleave(repeat_by, dim=0))
    controlnet_conditioning_image = controlnet_conditioning_image.to(device
        =device, dtype=dtype)
    if do_classifier_free_guidance:
        controlnet_conditioning_image = torch.cat([
            controlnet_conditioning_image] * 2)
    return controlnet_conditioning_image
