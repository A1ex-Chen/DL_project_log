def __call__(self, *inputs: np.ndarray) ->List[np.ndarray]:
    """
        Call the model with the given inputs.

        Args:
            *inputs (List[np.ndarray]): Input data to the model.

        Returns:
            (List[np.ndarray]): Model outputs.
        """
    infer_inputs = []
    input_format = inputs[0].dtype
    for i, x in enumerate(inputs):
        if x.dtype != self.np_input_formats[i]:
            x = x.astype(self.np_input_formats[i])
        infer_input = self.InferInput(self.input_names[i], [*x.shape], self
            .input_formats[i].replace('TYPE_', ''))
        infer_input.set_data_from_numpy(x)
        infer_inputs.append(infer_input)
    infer_outputs = [self.InferRequestedOutput(output_name) for output_name in
        self.output_names]
    outputs = self.triton_client.infer(model_name=self.endpoint, inputs=
        infer_inputs, outputs=infer_outputs)
    return [outputs.as_numpy(output_name).astype(input_format) for
        output_name in self.output_names]
