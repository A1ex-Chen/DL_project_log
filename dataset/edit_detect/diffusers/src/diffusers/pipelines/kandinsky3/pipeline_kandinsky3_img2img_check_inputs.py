def check_inputs(self, prompt, callback_steps, negative_prompt=None,
    prompt_embeds=None, negative_prompt_embeds=None,
    callback_on_step_end_tensor_inputs=None, attention_mask=None,
    negative_attention_mask=None):
    if callback_steps is not None and (not isinstance(callback_steps, int) or
        callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    if callback_on_step_end_tensor_inputs is not None and not all(k in self
        ._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
        raise ValueError(
            f'`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}'
            )
    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
            )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
    elif prompt is not None and (not isinstance(prompt, str) and not
        isinstance(prompt, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )
    if negative_prompt_embeds is not None and negative_attention_mask is None:
        raise ValueError(
            'Please provide `negative_attention_mask` along with `negative_prompt_embeds`'
            )
    if (negative_prompt_embeds is not None and negative_attention_mask is not
        None):
        if negative_prompt_embeds.shape[:2] != negative_attention_mask.shape:
            raise ValueError(
                f'`negative_prompt_embeds` and `negative_attention_mask` must have the same batch_size and token length when passed directly, but got: `negative_prompt_embeds` {negative_prompt_embeds.shape[:2]} != `negative_attention_mask` {negative_attention_mask.shape}.'
                )
    if prompt_embeds is not None and attention_mask is None:
        raise ValueError(
            'Please provide `attention_mask` along with `prompt_embeds`')
    if prompt_embeds is not None and attention_mask is not None:
        if prompt_embeds.shape[:2] != attention_mask.shape:
            raise ValueError(
                f'`prompt_embeds` and `attention_mask` must have the same batch_size and token length when passed directly, but got: `prompt_embeds` {prompt_embeds.shape[:2]} != `attention_mask` {attention_mask.shape}.'
                )
