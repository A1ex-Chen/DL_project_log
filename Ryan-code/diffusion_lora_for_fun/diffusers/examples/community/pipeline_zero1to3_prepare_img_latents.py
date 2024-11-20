def prepare_img_latents(self, image, batch_size, dtype, device, generator=
    None, do_classifier_free_guidance=False):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            assert image.shape[0
                ] == 3, 'Image outside a batch should be of shape (3, H, W)'
            image = image.unsqueeze(0)
        assert image.ndim == 4, 'Image must have 4 dimensions'
        if image.min() < -1 or image.max() > 1:
            raise ValueError('Image should be in [-1, 1] range')
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert('RGB'))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    image = image.to(device=device, dtype=dtype)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    if isinstance(generator, list):
        init_latents = [self.vae.encode(image[i:i + 1]).latent_dist.mode(
            generator[i]) for i in range(batch_size)]
        init_latents = torch.cat(init_latents, dim=0)
    else:
        init_latents = self.vae.encode(image).latent_dist.mode()
    if batch_size > init_latents.shape[0]:
        num_images_per_prompt = batch_size // init_latents.shape[0]
        bs_embed, emb_c, emb_h, emb_w = init_latents.shape
        init_latents = init_latents.unsqueeze(1)
        init_latents = init_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
        init_latents = init_latents.view(bs_embed * num_images_per_prompt,
            emb_c, emb_h, emb_w)
    init_latents = torch.cat([torch.zeros_like(init_latents), init_latents]
        ) if do_classifier_free_guidance else init_latents
    init_latents = init_latents.to(device=device, dtype=dtype)
    return init_latents
