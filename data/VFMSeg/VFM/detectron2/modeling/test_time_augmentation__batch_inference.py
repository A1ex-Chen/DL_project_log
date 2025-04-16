def _batch_inference(self, batched_inputs, detected_instances=None):
    """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
    if detected_instances is None:
        detected_instances = [None] * len(batched_inputs)
    outputs = []
    inputs, instances = [], []
    for idx, input, instance in zip(count(), batched_inputs, detected_instances
        ):
        inputs.append(input)
        instances.append(instance)
        if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
            outputs.extend(self.model.inference(inputs, instances if 
                instances[0] is not None else None, do_postprocess=False))
            inputs, instances = [], []
    return outputs
