def _check_and_fix_outputs(self, outputs: Vector) ->Vector:
    """Validate and optionally fix the outputs before returning them.

        Args:
            outputs (Vector): The output data.

        Returns:
            Vector: The validated and possibly fixed output data.
        """
    if self.output_vector_size and len(outputs) != self.output_vector_size:
        raise ValueError(
            f'Output vector size mismatch. Expected {self.output_vector_size}, got {len(outputs)}.'
            )
    self.output_batch_size = len(outputs)
    assert self.input_batch_size == self.output_batch_size, f'Input({self.input_batch_size}) and output({self.output_batch_size}) batch sizes do not match.'
    return np.array(outputs) if isinstance(outputs, torch.Tensor) else outputs
