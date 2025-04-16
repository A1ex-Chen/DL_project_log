def text2img(self, prompt: str=None, prompt_2: Optional[str]=None, height:
    Optional[int]=None, width: Optional[int]=None, num_inference_steps: int
    =50, timesteps: List[int]=None, denoising_start: Optional[float]=None,
    denoising_end: Optional[float]=None, guidance_scale: float=5.0,
    negative_prompt: Optional[str]=None, negative_prompt_2: Optional[str]=
    None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.Tensor]=None, ip_adapter_image: Optional[
    PipelineImageInput]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None,
    pooled_prompt_embeds: Optional[torch.Tensor]=None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor]=None, output_type:
    Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, guidance_rescale: float=0.0,
    original_size: Optional[Tuple[int, int]]=None, crops_coords_top_left:
    Tuple[int, int]=(0, 0), target_size: Optional[Tuple[int, int]]=None,
    clip_skip: Optional[int]=None, callback_on_step_end: Optional[Callable[
    [int, int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List
    [str]=['latents'], **kwargs):
    """
        Function invoked when calling pipeline for text-to-image.

        Refer to the documentation of the `__call__` method for parameter descriptions.
        """
    return self.__call__(prompt=prompt, prompt_2=prompt_2, height=height,
        width=width, num_inference_steps=num_inference_steps, timesteps=
        timesteps, denoising_start=denoising_start, denoising_end=
        denoising_end, guidance_scale=guidance_scale, negative_prompt=
        negative_prompt, negative_prompt_2=negative_prompt_2,
        num_images_per_prompt=num_images_per_prompt, eta=eta, generator=
        generator, latents=latents, ip_adapter_image=ip_adapter_image,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        output_type=output_type, return_dict=return_dict,
        cross_attention_kwargs=cross_attention_kwargs, guidance_rescale=
        guidance_rescale, original_size=original_size,
        crops_coords_top_left=crops_coords_top_left, target_size=
        target_size, clip_skip=clip_skip, callback_on_step_end=
        callback_on_step_end, callback_on_step_end_tensor_inputs=
        callback_on_step_end_tensor_inputs, **kwargs)
