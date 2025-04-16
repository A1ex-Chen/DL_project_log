@torch.no_grad()
def encode_prompt(self, prompt, do_classifier_free_guidance=True,
    num_images_per_prompt=1, device=None, negative_prompt=None,
    prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds:
    Optional[torch.Tensor]=None, _cut_context=False, attention_mask:
    Optional[torch.Tensor]=None, negative_attention_mask: Optional[torch.
    Tensor]=None):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask. Must provide if passing `prompt_embeds` directly.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated negative attention mask. Must provide if passing `negative_prompt_embeds` directly.
        """
    if prompt is not None and negative_prompt is not None:
        if type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
    if device is None:
        device = self._execution_device
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    max_length = 128
    if prompt_embeds is None:
        text_inputs = self.tokenizer(prompt, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=
            attention_mask)
        prompt_embeds = prompt_embeds[0]
        prompt_embeds, attention_mask = self._process_embeds(prompt_embeds,
            attention_mask, _cut_context)
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(2)
    if self.text_encoder is not None:
        dtype = self.text_encoder.dtype
    else:
        dtype = None
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
        seq_len, -1)
    attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        if negative_prompt is not None:
            uncond_input = self.tokenizer(uncond_tokens, padding=
                'max_length', max_length=128, truncation=True,
                return_attention_mask=True, return_tensors='pt')
            text_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(text_input_ids,
                attention_mask=negative_attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds[:, :
                prompt_embeds.shape[1]]
            negative_attention_mask = negative_attention_mask[:, :
                prompt_embeds.shape[1]]
            negative_prompt_embeds = (negative_prompt_embeds *
                negative_attention_mask.unsqueeze(2))
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_attention_mask = torch.zeros_like(attention_mask)
    if do_classifier_free_guidance:
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype,
            device=device)
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            negative_prompt_embeds = negative_prompt_embeds.repeat(1,
                num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
                num_images_per_prompt, seq_len, -1)
            negative_attention_mask = negative_attention_mask.repeat(
                num_images_per_prompt, 1)
    else:
        negative_prompt_embeds = None
        negative_attention_mask = None
    return (prompt_embeds, negative_prompt_embeds, attention_mask,
        negative_attention_mask)
