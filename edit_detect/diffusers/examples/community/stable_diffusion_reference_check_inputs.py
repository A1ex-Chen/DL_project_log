def check_inputs(self, prompt: Optional[Union[str, List[str]]], height: int,
    width: int, callback_steps: Optional[int], negative_prompt: Optional[
    str]=None, prompt_embeds: Optional[torch.Tensor]=None,
    negative_prompt_embeds: Optional[torch.Tensor]=None, ip_adapter_image:
    Optional[torch.Tensor]=None, ip_adapter_image_embeds: Optional[torch.
    Tensor]=None, callback_on_step_end_tensor_inputs: Optional[List[str]]=None
    ) ->None:
    """
        Check the validity of the input arguments for the diffusion model.

        Args:
            prompt (Optional[Union[str, List[str]]]): The prompt text or list of prompt texts.
            height (int): The height of the input image.
            width (int): The width of the input image.
            callback_steps (Optional[int]): The number of steps to perform the callback on.
            negative_prompt (Optional[str]): The negative prompt text.
            prompt_embeds (Optional[torch.Tensor]): The prompt embeddings.
            negative_prompt_embeds (Optional[torch.Tensor]): The negative prompt embeddings.
            ip_adapter_image (Optional[torch.Tensor]): The input adapter image.
            ip_adapter_image_embeds (Optional[torch.Tensor]): The input adapter image embeddings.
            callback_on_step_end_tensor_inputs (Optional[List[str]]): The list of tensor inputs to perform the callback on.

        Raises:
            ValueError: If `height` or `width` is not divisible by 8.
            ValueError: If `callback_steps` is not a positive integer.
            ValueError: If `callback_on_step_end_tensor_inputs` contains invalid tensor inputs.
            ValueError: If both `prompt` and `prompt_embeds` are provided.
            ValueError: If neither `prompt` nor `prompt_embeds` are provided.
            ValueError: If `prompt` is not of type `str` or `list`.
            ValueError: If both `negative_prompt` and `negative_prompt_embeds` are provided.
            ValueError: If both `prompt_embeds` and `negative_prompt_embeds` are provided and have different shapes.
            ValueError: If both `ip_adapter_image` and `ip_adapter_image_embeds` are provided.

        Returns:
            None
        """
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
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
    if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
        raise ValueError(
            'Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined.'
            )
