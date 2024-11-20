def encode_prompt(self, prompt, device, num_waveforms_per_prompt,
    do_classifier_free_guidance, transcription=None, negative_prompt=None,
    prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds:
    Optional[torch.Tensor]=None, generated_prompt_embeds: Optional[torch.
    Tensor]=None, negative_generated_prompt_embeds: Optional[torch.Tensor]=
    None, attention_mask: Optional[torch.LongTensor]=None,
    negative_attention_mask: Optional[torch.LongTensor]=None,
    max_new_tokens: Optional[int]=None):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            transcription (`str` or `List[str]`):
                transcription of text to speech
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, text embeddings will be computed from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            generated_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                The number of new tokens to generate with the GPT2 language model.
        Returns:
            prompt_embeds (`torch.Tensor`):
                Text embeddings from the Flan T5 model.
            attention_mask (`torch.LongTensor`):
                Attention mask to be applied to the `prompt_embeds`.
            generated_prompt_embeds (`torch.Tensor`):
                Text embeddings generated from the GPT2 langauge model.

        Example:

        ```python
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # Get text embedding vectors
        >>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        ...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
        ...     device="cuda",
        ...     do_classifier_free_guidance=True,
        ... )

        >>> # Pass text embeddings to pipeline for text-conditional audio generation
        >>> audio = pipe(
        ...     prompt_embeds=prompt_embeds,
        ...     attention_mask=attention_mask,
        ...     generated_prompt_embeds=generated_prompt_embeds,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ... ).audios[0]

        >>> # save generated audio sample
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```"""
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    tokenizers = [self.tokenizer, self.tokenizer_2]
    is_vits_text_encoder = isinstance(self.text_encoder_2, VitsModel)
    if is_vits_text_encoder:
        text_encoders = [self.text_encoder, self.text_encoder_2.text_encoder]
    else:
        text_encoders = [self.text_encoder, self.text_encoder_2]
    if prompt_embeds is None:
        prompt_embeds_list = []
        attention_mask_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            use_prompt = isinstance(tokenizer, (RobertaTokenizer,
                RobertaTokenizerFast, T5Tokenizer, T5TokenizerFast))
            text_inputs = tokenizer(prompt if use_prompt else transcription,
                padding='max_length' if isinstance(tokenizer, (
                RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer)) else
                True, max_length=tokenizer.model_max_length, truncation=
                True, return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = tokenizer(prompt, padding='longest',
                return_tensors='pt').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, 
                    tokenizer.model_max_length - 1:-1])
                logger.warning(
                    f'The following part of your input was truncated because {text_encoder.config.model_type} can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}'
                    )
            text_input_ids = text_input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if text_encoder.config.model_type == 'clap':
                prompt_embeds = text_encoder.get_text_features(text_input_ids,
                    attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[:, None, :]
                attention_mask = attention_mask.new_ones((batch_size, 1))
            elif is_vits_text_encoder:
                for text_input_id, text_attention_mask in zip(text_input_ids,
                    attention_mask):
                    for idx, phoneme_id in enumerate(text_input_id):
                        if phoneme_id == 0:
                            text_input_id[idx] = 182
                            text_attention_mask[idx] = 1
                            break
                prompt_embeds = text_encoder(text_input_ids, attention_mask
                    =attention_mask, padding_mask=attention_mask.unsqueeze(-1))
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = text_encoder(text_input_ids, attention_mask
                    =attention_mask)
                prompt_embeds = prompt_embeds[0]
            prompt_embeds_list.append(prompt_embeds)
            attention_mask_list.append(attention_mask)
        projection_output = self.projection_model(hidden_states=
            prompt_embeds_list[0], hidden_states_1=prompt_embeds_list[1],
            attention_mask=attention_mask_list[0], attention_mask_1=
            attention_mask_list[1])
        projected_prompt_embeds = projection_output.hidden_states
        projected_attention_mask = projection_output.attention_mask
        generated_prompt_embeds = self.generate_language_model(
            projected_prompt_embeds, attention_mask=
            projected_attention_mask, max_new_tokens=max_new_tokens)
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype,
        device=device)
    attention_mask = attention_mask.to(device=device
        ) if attention_mask is not None else torch.ones(prompt_embeds.shape
        [:2], dtype=torch.long, device=device)
    generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.
        language_model.dtype, device=device)
    bs_embed, seq_len, hidden_size = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt,
        seq_len, hidden_size)
    attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt)
    attention_mask = attention_mask.view(bs_embed *
        num_waveforms_per_prompt, seq_len)
    bs_embed, seq_len, hidden_size = generated_prompt_embeds.shape
    generated_prompt_embeds = generated_prompt_embeds.repeat(1,
        num_waveforms_per_prompt, 1)
    generated_prompt_embeds = generated_prompt_embeds.view(bs_embed *
        num_waveforms_per_prompt, seq_len, hidden_size)
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        negative_prompt_embeds_list = []
        negative_attention_mask_list = []
        max_length = prompt_embeds.shape[1]
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(uncond_tokens, padding='max_length',
                max_length=tokenizer.model_max_length if isinstance(
                tokenizer, (RobertaTokenizer, RobertaTokenizerFast,
                VitsTokenizer)) else max_length, truncation=True,
                return_tensors='pt')
            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)
            if text_encoder.config.model_type == 'clap':
                negative_prompt_embeds = text_encoder.get_text_features(
                    uncond_input_ids, attention_mask=negative_attention_mask)
                negative_prompt_embeds = negative_prompt_embeds[:, None, :]
                negative_attention_mask = negative_attention_mask.new_ones((
                    batch_size, 1))
            elif is_vits_text_encoder:
                negative_prompt_embeds = torch.zeros(batch_size, tokenizer.
                    model_max_length, text_encoder.config.hidden_size).to(dtype
                    =self.text_encoder_2.dtype, device=device)
                negative_attention_mask = torch.zeros(batch_size, tokenizer
                    .model_max_length).to(dtype=self.text_encoder_2.dtype,
                    device=device)
            else:
                negative_prompt_embeds = text_encoder(uncond_input_ids,
                    attention_mask=negative_attention_mask)
                negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            negative_attention_mask_list.append(negative_attention_mask)
        projection_output = self.projection_model(hidden_states=
            negative_prompt_embeds_list[0], hidden_states_1=
            negative_prompt_embeds_list[1], attention_mask=
            negative_attention_mask_list[0], attention_mask_1=
            negative_attention_mask_list[1])
        negative_projected_prompt_embeds = projection_output.hidden_states
        negative_projected_attention_mask = projection_output.attention_mask
        negative_generated_prompt_embeds = self.generate_language_model(
            negative_projected_prompt_embeds, attention_mask=
            negative_projected_attention_mask, max_new_tokens=max_new_tokens)
    if do_classifier_free_guidance:
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.
            text_encoder_2.dtype, device=device)
        negative_attention_mask = negative_attention_mask.to(device=device
            ) if negative_attention_mask is not None else torch.ones(
            negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
        negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
            dtype=self.language_model.dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_waveforms_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_waveforms_per_prompt, seq_len, -1)
        negative_attention_mask = negative_attention_mask.repeat(1,
            num_waveforms_per_prompt)
        negative_attention_mask = negative_attention_mask.view(batch_size *
            num_waveforms_per_prompt, seq_len)
        seq_len = negative_generated_prompt_embeds.shape[1]
        negative_generated_prompt_embeds = (negative_generated_prompt_embeds
            .repeat(1, num_waveforms_per_prompt, 1))
        negative_generated_prompt_embeds = (negative_generated_prompt_embeds
            .view(batch_size * num_waveforms_per_prompt, seq_len, -1))
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        attention_mask = torch.cat([negative_attention_mask, attention_mask])
        generated_prompt_embeds = torch.cat([
            negative_generated_prompt_embeds, generated_prompt_embeds])
    return prompt_embeds, attention_mask, generated_prompt_embeds
