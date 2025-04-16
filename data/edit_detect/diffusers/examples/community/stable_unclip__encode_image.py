def _encode_image(self, image, device, num_images_per_prompt,
    do_classifier_free_guidance):
    image = image.to(device=device)
    image_embeddings = image
    image_embeddings = image_embeddings.unsqueeze(1)
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed *
        num_images_per_prompt, seq_len, -1)
    if do_classifier_free_guidance:
        uncond_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([uncond_embeddings, image_embeddings])
    return image_embeddings
