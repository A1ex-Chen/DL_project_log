def _encode_prompt(self, prompt, device, do_classifier_free_guidance,
    negative_prompt):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_length=
        True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer(prompt, padding='longest',
        return_tensors='pt').input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
        ] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.
            tokenizer.model_max_length - 1:-1])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
    text_encoder_out = self.text_encoder(text_input_ids.to(device),
        output_hidden_states=True)
    text_embeddings = text_encoder_out.hidden_states[-1]
    text_pooler_out = text_encoder_out.pooler_output
    if do_classifier_free_guidance:
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
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_length=True,
            return_tensors='pt')
        uncond_encoder_out = self.text_encoder(uncond_input.input_ids.to(
            device), output_hidden_states=True)
        uncond_embeddings = uncond_encoder_out.hidden_states[-1]
        uncond_pooler_out = uncond_encoder_out.pooler_output
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_pooler_out = torch.cat([uncond_pooler_out, text_pooler_out])
    return text_embeddings, text_pooler_out
