def check_inputs(self, negative_prompt=None, editing_prompt_embeddings=None,
    negative_prompt_embeds=None, callback_on_step_end_tensor_inputs=None):
    if callback_on_step_end_tensor_inputs is not None and not all(k in self
        ._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
        raise ValueError(
            f'`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}'
            )
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    if (editing_prompt_embeddings is not None and negative_prompt_embeds is not
        None):
        if editing_prompt_embeddings.shape != negative_prompt_embeds.shape:
            raise ValueError(
                f'`editing_prompt_embeddings` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `editing_prompt_embeddings` {editing_prompt_embeddings.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )
