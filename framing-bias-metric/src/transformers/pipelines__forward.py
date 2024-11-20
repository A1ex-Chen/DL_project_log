def _forward(self, inputs, return_tensors=False):
    """
        Internal framework specific forward dispatching

        Args:
            inputs: dict holding all the keyword arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array

        Returns:
            Numpy array
        """
    with self.device_placement():
        if self.framework == 'tf':
            predictions = self.model(inputs.data, training=False)[0]
        else:
            with torch.no_grad():
                inputs = self.ensure_tensor_on_device(**inputs)
                predictions = self.model(**inputs)[0].cpu()
    if return_tensors:
        return predictions
    else:
        return predictions.numpy()
