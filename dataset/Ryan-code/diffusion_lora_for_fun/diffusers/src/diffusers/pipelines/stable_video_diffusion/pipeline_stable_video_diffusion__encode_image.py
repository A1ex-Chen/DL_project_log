def _encode_image(self, image: PipelineImageInput, device: Union[str, torch
    .device], num_videos_per_prompt: int, do_classifier_free_guidance: bool
    ) ->torch.Tensor:
    dtype = next(self.image_encoder.parameters()).dtype
    if not isinstance(image, torch.Tensor):
        image = self.video_processor.pil_to_numpy(image)
        image = self.video_processor.numpy_to_pt(image)
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
    image = self.feature_extractor(images=image, do_normalize=True,
        do_center_crop=False, do_resize=False, do_rescale=False,
        return_tensors='pt').pixel_values
    image = image.to(device=device, dtype=dtype)
    image_embeddings = self.image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed *
        num_videos_per_prompt, seq_len, -1)
    if do_classifier_free_guidance:
        negative_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_image_embeddings,
            image_embeddings])
    return image_embeddings
