@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt: Union[str, List[str]], image: Union[torch.Tensor,
    List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], strength:
    float=0.3, negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: int=1, num_inference_steps: int=25, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    guidance_scale: float=4.0, output_type: Optional[str]='pt', return_dict:
    bool=True):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `emb`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added.
            emb (`torch.Tensor`):
                The image embedding.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"`
                (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """
    if isinstance(prompt, str):
        prompt = [prompt]
    elif not isinstance(prompt, list):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
    elif not isinstance(negative_prompt, list) and negative_prompt is not None:
        raise ValueError(
            f'`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}'
            )
    if negative_prompt is not None:
        prompt = prompt + negative_prompt
        negative_prompt = 2 * negative_prompt
    device = self._execution_device
    batch_size = len(prompt)
    batch_size = batch_size * num_images_per_prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance,
        negative_prompt)
    if not isinstance(image, List):
        image = [image]
    if isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    if isinstance(image, torch.Tensor) and image.ndim == 2:
        image_embeds = image.repeat_interleave(num_images_per_prompt, dim=0)
    elif isinstance(image, torch.Tensor) and image.ndim != 4:
        raise ValueError(
            f' if pass `image` as pytorch tensor, or a list of pytorch tensor, please make sure each tensor has shape [batch_size, channels, height, width], currently {image[0].unsqueeze(0).shape}'
            )
    else:
        image_embeds = self._encode_image(image, device, num_images_per_prompt)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = image_embeds
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size)
    latents = self.prepare_latents(latents, latent_timestep, batch_size //
        num_images_per_prompt, num_images_per_prompt, prompt_embeds.dtype,
        device, generator)
    for i, t in enumerate(self.progress_bar(timesteps)):
        latent_model_input = torch.cat([latents] * 2
            ) if do_classifier_free_guidance else latents
        predicted_image_embedding = self.prior(latent_model_input, timestep
            =t, proj_embedding=prompt_embeds, encoder_hidden_states=
            text_encoder_hidden_states, attention_mask=text_mask
            ).predicted_image_embedding
        if do_classifier_free_guidance:
            (predicted_image_embedding_uncond, predicted_image_embedding_text
                ) = predicted_image_embedding.chunk(2)
            predicted_image_embedding = (predicted_image_embedding_uncond +
                guidance_scale * (predicted_image_embedding_text -
                predicted_image_embedding_uncond))
        if i + 1 == timesteps.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = timesteps[i + 1]
        latents = self.scheduler.step(predicted_image_embedding, timestep=t,
            sample=latents, generator=generator, prev_timestep=prev_timestep
            ).prev_sample
    latents = self.prior.post_process_latents(latents)
    image_embeddings = latents
    if negative_prompt is None:
        zero_embeds = self.get_zero_embed(latents.shape[0], device=latents.
            device)
    else:
        image_embeddings, zero_embeds = image_embeddings.chunk(2)
    self.maybe_free_model_hooks()
    if output_type not in ['pt', 'np']:
        raise ValueError(
            f'Only the output types `pt` and `np` are supported not output_type={output_type}'
            )
    if output_type == 'np':
        image_embeddings = image_embeddings.cpu().numpy()
        zero_embeds = zero_embeds.cpu().numpy()
    if not return_dict:
        return image_embeddings, zero_embeds
    return KandinskyPriorPipelineOutput(image_embeds=image_embeddings,
        negative_image_embeds=zero_embeds)
