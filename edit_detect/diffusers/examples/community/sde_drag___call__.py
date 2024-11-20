@torch.no_grad()
def __call__(self, prompt: str, image: PIL.Image.Image, mask_image: PIL.
    Image.Image, source_points: List[List[int]], target_points: List[List[
    int]], t0: Optional[float]=0.6, steps: Optional[int]=200, step_size:
    Optional[int]=2, image_scale: Optional[float]=0.3, adapt_radius:
    Optional[int]=5, min_lora_scale: Optional[float]=0.5, generator:
    Optional[torch.Generator]=None):
    """
        Function invoked when calling the pipeline for image editing.
        Args:
            prompt (`str`, *required*):
                The prompt to guide the image editing.
            image (`PIL.Image.Image`, *required*):
                Which will be edited, parts of the image will be masked out with `mask_image` and edited
                according to `prompt`.
            mask_image (`PIL.Image.Image`, *required*):
                To mask `image`. White pixels in the mask will be edited, while black pixels will be preserved.
            source_points (`List[List[int]]`, *required*):
                Used to mark the starting positions of drag editing in the image, with each pixel represented as a
                `List[int]` of length 2.
            target_points (`List[List[int]]`, *required*):
                Used to mark the target positions of drag editing in the image, with each pixel represented as a
                `List[int]` of length 2.
            t0 (`float`, *optional*, defaults to 0.6):
                The time parameter. Higher t0 improves the fidelity while lowering the faithfulness of the edited images
                and vice versa.
            steps (`int`, *optional*, defaults to 200):
                The number of sampling iterations.
            step_size (`int`, *optional*, defaults to 2):
                The drag diatance of each drag step.
            image_scale (`float`, *optional*, defaults to 0.3):
                To avoid duplicating the content, use image_scale to perturbs the source.
            adapt_radius (`int`, *optional*, defaults to 5):
                The size of the region for copy and paste operations during each step of the drag process.
            min_lora_scale (`float`, *optional*, defaults to 0.5):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                min_lora_scale specifies the minimum LoRA scale during the image drag-editing process.
            generator ('torch.Generator', *optional*, defaults to None):
                To make generation deterministic(https://pytorch.org/docs/stable/generated/torch.Generator.html).
        Examples:
        ```py
        >>> import PIL
        >>> import torch
        >>> from diffusers import DDIMScheduler, DiffusionPipeline

        >>> # Load the pipeline
        >>> model_path = "runwayml/stable-diffusion-v1-5"
        >>> scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        >>> pipe = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, custom_pipeline="sde_drag")
        >>> pipe.to('cuda')

        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> # If not training LoRA, please avoid using torch.float16
        >>> # pipe.to(torch.float16)

        >>> # Provide prompt, image, mask image, and the starting and target points for drag editing.
        >>> prompt = "prompt of the image"
        >>> image = PIL.Image.open('/path/to/image')
        >>> mask_image = PIL.Image.open('/path/to/mask_image')
        >>> source_points = [[123, 456]]
        >>> target_points = [[234, 567]]

        >>> # train_lora is optional, and in most cases, using train_lora can better preserve consistency with the original image.
        >>> pipe.train_lora(prompt, image)

        >>> output = pipe(prompt, image, mask_image, source_points, target_points)
        >>> output_image = PIL.Image.fromarray(output)
        >>> output_image.save("./output.png")
        ```
        """
    self.scheduler.set_timesteps(steps)
    noise_scale = (1 - image_scale ** 2) ** 0.5
    text_embeddings = self._get_text_embed(prompt)
    uncond_embeddings = self._get_text_embed([''])
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latent = self._get_img_latent(image)
    mask = mask_image.resize((latent.shape[3], latent.shape[2]))
    mask = torch.tensor(np.array(mask))
    mask = mask.unsqueeze(0).expand_as(latent).to(self.device)
    source_points = torch.tensor(source_points).div(torch.tensor([8]),
        rounding_mode='trunc')
    target_points = torch.tensor(target_points).div(torch.tensor([8]),
        rounding_mode='trunc')
    distance = target_points - source_points
    distance_norm_max = torch.norm(distance.float(), dim=1, keepdim=True).max()
    if distance_norm_max <= step_size:
        drag_num = 1
    else:
        drag_num = distance_norm_max.div(torch.tensor([step_size]),
            rounding_mode='trunc')
        if (distance_norm_max / drag_num - step_size).abs() > (
            distance_norm_max / (drag_num + 1) - step_size).abs():
            drag_num += 1
    latents = []
    for i in tqdm(range(int(drag_num)), desc='SDE Drag'):
        source_new = source_points + (i / drag_num * distance).to(torch.int)
        target_new = source_points + ((i + 1) / drag_num * distance).to(torch
            .int)
        latent, noises, hook_latents, lora_scales, cfg_scales = self._forward(
            latent, steps, t0, min_lora_scale, text_embeddings, generator)
        latent = self._copy_and_paste(latent, source_new, target_new,
            adapt_radius, latent.shape[2] - 1, latent.shape[3] - 1,
            image_scale, noise_scale, generator)
        latent = self._backward(latent, mask, steps, t0, noises,
            hook_latents, lora_scales, cfg_scales, text_embeddings, generator)
        latents.append(latent)
    result_image = 1 / 0.18215 * latents[-1]
    with torch.no_grad():
        result_image = self.vae.decode(result_image).sample
    result_image = (result_image / 2 + 0.5).clamp(0, 1)
    result_image = result_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    result_image = (result_image * 255).astype(np.uint8)
    return result_image
