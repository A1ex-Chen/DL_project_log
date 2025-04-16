def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]
    ) ->int:
    """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (:obj:`dict`): The model inputs.

        Returns:
            :obj:`int`: The total number of tokens.
        """
    token_inputs = [tensor for key, tensor in input_dict.items() if 'input' in
        key]
    if token_inputs:
        return sum([token_input.numel() for token_input in token_inputs])
    else:
        warnings.warn(
            'Could not estimate the number of tokens of the input, floating-point operations will not be computed'
            )
        return 0
