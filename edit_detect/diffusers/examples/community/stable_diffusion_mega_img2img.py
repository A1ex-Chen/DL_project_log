@torch.no_grad()
def img2img(self, prompt: Union[str, List[str]], image: Union[torch.Tensor,
    PIL.Image.Image], strength: float=0.8, num_inference_steps: Optional[
    int]=50, guidance_scale: Optional[float]=7.5, negative_prompt: Optional
    [Union[str, List[str]]]=None, num_images_per_prompt: Optional[int]=1,
    eta: Optional[float]=0.0, generator: Optional[torch.Generator]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, callback:
    Optional[Callable[[int, int, torch.Tensor], None]]=None, callback_steps:
    int=1, **kwargs):
    return StableDiffusionImg2ImgPipeline(**self.components)(prompt=prompt,
        image=image, strength=strength, num_inference_steps=
        num_inference_steps, guidance_scale=guidance_scale, negative_prompt
        =negative_prompt, num_images_per_prompt=num_images_per_prompt, eta=
        eta, generator=generator, output_type=output_type, return_dict=
        return_dict, callback=callback, callback_steps=callback_steps)
