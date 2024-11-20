def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer:
    'PreTrainedTokenizer'):
    """
        Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
        be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or if the textual inversion token is a single vector, the input prompt is returned.

        Parameters:
            prompt (`str` or list of `str`):
                The prompt or prompts to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str` or list of `str`: The converted prompt
        """
    if not isinstance(prompt, List):
        prompts = [prompt]
    else:
        prompts = prompt
    prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]
    if not isinstance(prompt, List):
        return prompts[0]
    return prompts
