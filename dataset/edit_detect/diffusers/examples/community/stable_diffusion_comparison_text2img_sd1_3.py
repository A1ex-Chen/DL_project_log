@torch.no_grad()
def text2img_sd1_3(self, prompt: Union[str, List[str]], height: int=512,
    width: int=512, num_inference_steps: int=50, guidance_scale: float=7.5,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[torch.Generator]=None, latents: Optional[torch.Tensor]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, callback:
    Optional[Callable[[int, int, torch.Tensor], None]]=None, callback_steps:
    int=1, **kwargs):
    return self.pipe3(prompt=prompt, height=height, width=width,
        num_inference_steps=num_inference_steps, guidance_scale=
        guidance_scale, negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt, eta=eta, generator=
        generator, latents=latents, output_type=output_type, return_dict=
        return_dict, callback=callback, callback_steps=callback_steps, **kwargs
        )
