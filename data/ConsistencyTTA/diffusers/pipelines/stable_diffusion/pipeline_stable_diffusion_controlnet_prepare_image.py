def prepare_image(self, image, width, height, batch_size,
    num_images_per_prompt, device, dtype, do_classifier_free_guidance):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        if isinstance(image[0], PIL.Image.Image):
            images = []
            for image_ in image:
                image_ = image_.convert('RGB')
                image_ = image_.resize((width, height), resample=
                    PIL_INTERPOLATION['lanczos'])
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)
            image = images
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
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
