def encode_prompt(self, device, num_images_per_prompt, enable_edit_guidance,
    negative_prompt=None, editing_prompt=None, negative_prompt_embeds:
    Optional[torch.Tensor]=None, editing_prompt_embeds: Optional[torch.
    Tensor]=None, lora_scale: Optional[float]=None, clip_skip: Optional[int
    ]=None):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            enable_edit_guidance (`bool`):
                whether to perform any editing or reconstruct the input image instead
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            editing_prompt (`str` or `List[str]`, *optional*):
                Editing prompt(s) to be encoded. If not defined, one has to pass `editing_prompt_embeds` instead.
            editing_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
    if lora_scale is not None and isinstance(self, LoraLoaderMixin):
        self._lora_scale = lora_scale
        if not USE_PEFT_BACKEND:
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
        else:
            scale_lora_layers(self.text_encoder, lora_scale)
    batch_size = self.batch_size
    num_edit_tokens = None
    if negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but exoected{batch_size} based on the input images. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        if isinstance(self, TextualInversionLoaderMixin):
            uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.
                tokenizer)
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt')
        if hasattr(self.text_encoder.config, 'use_attention_mask'
            ) and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None
        negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.
            to(device), attention_mask=attention_mask)
        negative_prompt_embeds = negative_prompt_embeds[0]
    if self.text_encoder is not None:
        prompt_embeds_dtype = self.text_encoder.dtype
    elif self.unet is not None:
        prompt_embeds_dtype = self.unet.dtype
    else:
        prompt_embeds_dtype = negative_prompt_embeds.dtype
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=
        prompt_embeds_dtype, device=device)
    if enable_edit_guidance:
        if editing_prompt_embeds is None:
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            max_length = negative_prompt_embeds.shape[1]
            text_inputs = self.tokenizer([x for item in editing_prompt for
                x in repeat(item, batch_size)], padding='max_length',
                max_length=max_length, truncation=True, return_tensors='pt',
                return_length=True)
            num_edit_tokens = text_inputs.length - 2
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer([x for item in editing_prompt for
                x in repeat(item, batch_size)], padding='longest',
                return_tensors='pt').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[
                    :, self.tokenizer.model_max_length - 1:-1])
                logger.warning(
                    f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                    )
            if hasattr(self.text_encoder.config, 'use_attention_mask'
                ) and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            if clip_skip is None:
                editing_prompt_embeds = self.text_encoder(text_input_ids.to
                    (device), attention_mask=attention_mask)
                editing_prompt_embeds = editing_prompt_embeds[0]
            else:
                editing_prompt_embeds = self.text_encoder(text_input_ids.to
                    (device), attention_mask=attention_mask,
                    output_hidden_states=True)
                editing_prompt_embeds = editing_prompt_embeds[-1][-(
                    clip_skip + 1)]
                editing_prompt_embeds = (self.text_encoder.text_model.
                    final_layer_norm(editing_prompt_embeds))
        editing_prompt_embeds = editing_prompt_embeds.to(dtype=
            negative_prompt_embeds.dtype, device=device)
        bs_embed_edit, seq_len, _ = editing_prompt_embeds.shape
        editing_prompt_embeds = editing_prompt_embeds.to(dtype=
            negative_prompt_embeds.dtype, device=device)
        editing_prompt_embeds = editing_prompt_embeds.repeat(1,
            num_images_per_prompt, 1)
        editing_prompt_embeds = editing_prompt_embeds.view(bs_embed_edit *
            num_images_per_prompt, seq_len, -1)
    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=
        prompt_embeds_dtype, device=device)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1,
        num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
        num_images_per_prompt, seq_len, -1)
    if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
        unscale_lora_layers(self.text_encoder, lora_scale)
    return editing_prompt_embeds, negative_prompt_embeds, num_edit_tokens
