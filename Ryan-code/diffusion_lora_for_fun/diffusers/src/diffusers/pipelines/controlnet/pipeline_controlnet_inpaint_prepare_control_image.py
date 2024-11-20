def prepare_control_image(self, image, width, height, batch_size,
    num_images_per_prompt, device, dtype, crops_coords, resize_mode,
    do_classifier_free_guidance=False, guess_mode=False):
    image = self.control_image_processor.preprocess(image, height=height,
        width=width, crops_coords=crops_coords, resize_mode=resize_mode).to(
        dtype=torch.float32)
    image_batch_size = image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        repeat_by = num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)
    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)
    return image
