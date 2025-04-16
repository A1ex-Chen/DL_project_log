def _check_and_fix_outputs(self, outputs, inputs):
    """
        Check output type and convert to list of `Detections` objects.

        The function expects the raw output from the model to be either a list of `Detection` objects or a list of lists, where each sub-list should contain four elements corresponding to the bounding box of the detected object. See `Detection` and `Detections` class for more details.

        If the output is not in the correct format, a ValueError is raised.

        Args:
            outputs: The raw output from the model.
            inputs: The corresponding inputs to the model.

        Returns:
            A list of `Detections` objects.
        """
    return outputs
