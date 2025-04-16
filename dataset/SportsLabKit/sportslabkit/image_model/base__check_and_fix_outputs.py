def _check_and_fix_outputs(self, outputs):
    """
        Check output type and convert to a 2D numpy array.

        The function expects the raw output from the model to be either an iterable of embeddings of the same length as the input, or a single embedding. If the output is not in the correct format, a ValueError is raised.

        If the output is not in the correct format, a ValueError is raised.

        Args:
            outputs: The raw output from the model.

        Returns:
            A list of embeddings.
        """
    if not isinstance(outputs, Iterable):
        raise ValueError(f'Model output is not iterable. Got {type(outputs)}')
    if isinstance(outputs[0], (int, float)):
        outputs = [[output] for output in outputs]
    if isinstance(outputs[0], torch.Tensor):
        outputs = [output.detach().cpu().numpy() for output in outputs]
    return np.stack(outputs)
