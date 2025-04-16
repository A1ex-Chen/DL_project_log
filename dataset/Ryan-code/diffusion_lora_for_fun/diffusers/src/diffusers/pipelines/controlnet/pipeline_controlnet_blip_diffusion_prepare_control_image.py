def prepare_control_image(self, image, width, height, batch_size,
    num_images_per_prompt, device, dtype, do_classifier_free_guidance=False):
    image = self.image_processor.preprocess(image, size={'width': width,
        'height': height}, do_rescale=True, do_center_crop=False,
        do_normalize=False, return_tensors='pt')['pixel_values'].to(device)
    image_batch_size = image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        repeat_by = num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)
    if do_classifier_free_guidance:
        image = torch.cat([image] * 2)
    return image
