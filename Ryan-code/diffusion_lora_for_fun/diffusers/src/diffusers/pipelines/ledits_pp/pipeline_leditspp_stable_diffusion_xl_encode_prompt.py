def encode_prompt(self, device: Optional[torch.device]=None,
    num_images_per_prompt: int=1, negative_prompt: Optional[str]=None,
    negative_prompt_2: Optional[str]=None, negative_prompt_embeds: Optional
    [torch.Tensor]=None, negative_pooled_prompt_embeds: Optional[torch.
    Tensor]=None, lora_scale: Optional[float]=None, clip_skip: Optional[int
    ]=None, enable_edit_guidance: bool=True, editing_prompt: Optional[str]=
    None, editing_prompt_embeds: Optional[torch.Tensor]=None,
    editing_pooled_prompt_embeds: Optional[torch.Tensor]=None) ->object:
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            enable_edit_guidance (`bool`):
                Whether to guide towards an editing prompt or not.
            editing_prompt (`str` or `List[str]`, *optional*):
                Editing prompt(s) to be encoded. If not defined and 'enable_edit_guidance' is True, one has to pass
                `editing_prompt_embeds` instead.
            editing_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided and 'enable_edit_guidance' is True, editing_prompt_embeds will be generated from
                `editing_prompt` input argument.
            editing_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated edit pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled editing_pooled_prompt_embeds will be generated from `editing_prompt`
                input argument.
        """
    device = device or self._execution_device
    if lora_scale is not None and isinstance(self,
        StableDiffusionXLLoraLoaderMixin):
        self._lora_scale = lora_scale
        if self.text_encoder is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
            else:
                scale_lora_layers(self.text_encoder_2, lora_scale)
    batch_size = self.batch_size
    tokenizers = [self.tokenizer, self.tokenizer_2
        ] if self.tokenizer is not None else [self.tokenizer_2]
    text_encoders = [self.text_encoder, self.text_encoder_2
        ] if self.text_encoder is not None else [self.text_encoder_2]
    num_edit_tokens = 0
    zero_out_negative_prompt = (negative_prompt is None and self.config.
        force_zeros_for_empty_prompt)
    if negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ''
        negative_prompt_2 = negative_prompt_2 or negative_prompt
        negative_prompt = batch_size * [negative_prompt] if isinstance(
            negative_prompt, str) else negative_prompt
        negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(
            negative_prompt_2, str) else negative_prompt_2
        uncond_tokens: List[str]
        if batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but image inversion  has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of the input images.'
                )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]
        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens,
            tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                negative_prompt = self.maybe_convert_prompt(negative_prompt,
                    tokenizer)
            uncond_input = tokenizer(negative_prompt, padding='max_length',
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors='pt')
            negative_prompt_embeds = text_encoder(uncond_input.input_ids.to
                (device), output_hidden_states=True)
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            negative_prompt_embeds_list.append(negative_prompt_embeds)
        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list,
            dim=-1)
        if zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(negative_prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(
                negative_pooled_prompt_embeds)
    if enable_edit_guidance and editing_prompt_embeds is None:
        editing_prompt_2 = editing_prompt
        editing_prompts = [editing_prompt, editing_prompt_2]
        edit_prompt_embeds_list = []
        for editing_prompt, tokenizer, text_encoder in zip(editing_prompts,
            tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                editing_prompt = self.maybe_convert_prompt(editing_prompt,
                    tokenizer)
            max_length = negative_prompt_embeds.shape[1]
            edit_concepts_input = tokenizer(editing_prompt, padding=
                'max_length', max_length=max_length, truncation=True,
                return_tensors='pt', return_length=True)
            num_edit_tokens = edit_concepts_input.length - 2
            edit_concepts_embeds = text_encoder(edit_concepts_input.
                input_ids.to(device), output_hidden_states=True)
            editing_pooled_prompt_embeds = edit_concepts_embeds[0]
            if clip_skip is None:
                edit_concepts_embeds = edit_concepts_embeds.hidden_states[-2]
            else:
                edit_concepts_embeds = edit_concepts_embeds.hidden_states[-
                    (clip_skip + 2)]
            edit_prompt_embeds_list.append(edit_concepts_embeds)
        edit_concepts_embeds = torch.concat(edit_prompt_embeds_list, dim=-1)
    elif not enable_edit_guidance:
        edit_concepts_embeds = None
        editing_pooled_prompt_embeds = None
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.
        text_encoder_2.dtype, device=device)
    bs_embed, seq_len, _ = negative_prompt_embeds.shape
    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.
        text_encoder_2.dtype, device=device)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1,
        num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
        num_images_per_prompt, seq_len, -1)
    if enable_edit_guidance:
        bs_embed_edit, seq_len, _ = edit_concepts_embeds.shape
        edit_concepts_embeds = edit_concepts_embeds.to(dtype=self.
            text_encoder_2.dtype, device=device)
        edit_concepts_embeds = edit_concepts_embeds.repeat(1,
            num_images_per_prompt, 1)
        edit_concepts_embeds = edit_concepts_embeds.view(bs_embed_edit *
            num_images_per_prompt, seq_len, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1,
        num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
    if enable_edit_guidance:
        editing_pooled_prompt_embeds = editing_pooled_prompt_embeds.repeat(
            1, num_images_per_prompt).view(bs_embed_edit *
            num_images_per_prompt, -1)
    if self.text_encoder is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin
            ) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder, lora_scale)
    if self.text_encoder_2 is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin
            ) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder_2, lora_scale)
    return (negative_prompt_embeds, edit_concepts_embeds,
        negative_pooled_prompt_embeds, editing_pooled_prompt_embeds,
        num_edit_tokens)
