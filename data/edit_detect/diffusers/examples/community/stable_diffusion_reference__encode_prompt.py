def _encode_prompt(self, prompt: Union[str, List[str]], device: torch.
    device, num_images_per_prompt: int, do_classifier_free_guidance: bool,
    negative_prompt: Optional[Union[str, List[str]]]=None, prompt_embeds:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, lora_scale: Optional[float]=None, **kwargs) ->torch.Tensor:
    """
        Encodes the prompt into embeddings.

        Args:
            prompt (Union[str, List[str]]): The prompt text or a list of prompt texts.
            device (torch.device): The device to use for encoding.
            num_images_per_prompt (int): The number of images per prompt.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.
            negative_prompt (Optional[Union[str, List[str]]], optional): The negative prompt text or a list of negative prompt texts. Defaults to None.
            prompt_embeds (Optional[torch.Tensor], optional): The prompt embeddings. Defaults to None.
            negative_prompt_embeds (Optional[torch.Tensor], optional): The negative prompt embeddings. Defaults to None.
            lora_scale (Optional[float], optional): The LoRA scale. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The encoded prompt embeddings.
        """
    deprecation_message = (
        '`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple.'
        )
    deprecate('_encode_prompt()', '1.0.0', deprecation_message,
        standard_warn=False)
    prompt_embeds_tuple = self.encode_prompt(prompt=prompt, device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, lora_scale=
        lora_scale, **kwargs)
    prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    return prompt_embeds
