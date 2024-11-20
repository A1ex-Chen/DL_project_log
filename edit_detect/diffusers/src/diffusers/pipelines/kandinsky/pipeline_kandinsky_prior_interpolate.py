@torch.no_grad()
@replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)
def interpolate(self, images_and_prompts: List[Union[str, PIL.Image.Image,
    torch.Tensor]], weights: List[float], num_images_per_prompt: int=1,
    num_inference_steps: int=25, generator: Optional[Union[torch.Generator,
    List[torch.Generator]]]=None, latents: Optional[torch.Tensor]=None,
    negative_prior_prompt: Optional[str]=None, negative_prompt: str='',
    guidance_scale: float=4.0, device=None):
    """
        Function invoked when using the prior pipeline for interpolation.

        Args:
            images_and_prompts (`List[Union[str, PIL.Image.Image, torch.Tensor]]`):
                list of prompts and images to guide the image generation.
            weights: (`List[float]`):
                list of weights for each condition in `images_and_prompts`
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            negative_prior_prompt (`str`, *optional*):
                The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """
    device = device or self.device
    if len(images_and_prompts) != len(weights):
        raise ValueError(
            f'`images_and_prompts` contains {len(images_and_prompts)} items and `weights` contains {len(weights)} items - they should be lists of same length'
            )
    image_embeddings = []
    for cond, weight in zip(images_and_prompts, weights):
        if isinstance(cond, str):
            image_emb = self(cond, num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt, generator=
                generator, latents=latents, negative_prompt=
                negative_prior_prompt, guidance_scale=guidance_scale
                ).image_embeds
        elif isinstance(cond, (PIL.Image.Image, torch.Tensor)):
            if isinstance(cond, PIL.Image.Image):
                cond = self.image_processor(cond, return_tensors='pt'
                    ).pixel_values[0].unsqueeze(0).to(dtype=self.
                    image_encoder.dtype, device=device)
            image_emb = self.image_encoder(cond)['image_embeds']
        else:
            raise ValueError(
                f'`images_and_prompts` can only contains elements to be of type `str`, `PIL.Image.Image` or `torch.Tensor`  but is {type(cond)}'
                )
        image_embeddings.append(image_emb * weight)
    image_emb = torch.cat(image_embeddings).sum(dim=0, keepdim=True)
    out_zero = self(negative_prompt, num_inference_steps=
        num_inference_steps, num_images_per_prompt=num_images_per_prompt,
        generator=generator, latents=latents, negative_prompt=
        negative_prior_prompt, guidance_scale=guidance_scale)
    zero_image_emb = (out_zero.negative_image_embeds if negative_prompt ==
        '' else out_zero.image_embeds)
    return KandinskyPriorPipelineOutput(image_embeds=image_emb,
        negative_image_embeds=zero_image_emb)
