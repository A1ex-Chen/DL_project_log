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
    if len(outputs) != len(inputs):
        raise ValueError(
            f'Length of outputs does not match length of inputs. Got {len(outputs)} outputs and {len(inputs)} inputs.'
            )
    if isinstance(outputs[0], Detections):
        return outputs
    check_1 = not isinstance(outputs, (list, tuple)) or not isinstance(outputs
        [0], (list, tuple))
    check_2 = isinstance(outputs[0][0], int) or isinstance(outputs[0][0], float
        )
    if check_1 or check_2:
        raise ValueError(
            "The model's output should be a list of list of Detection objects or a compatible object."
            )
    list_of_detections = []
    for preds, image in zip(outputs, inputs):
        dets = []
        for pred in preds:
            if pred == {}:
                continue
            det = convert_to_detection(pred)
            dets.append(det)
        list_of_detections.append(Detections(dets, image))
    if not list_of_detections:
        raise ValueError("Empty list of detections. Check your model's output."
            )
    if len(list_of_detections) != len(inputs):
        raise ValueError(
            'Length of detections does not match length of inputs.')
    return list_of_detections
