def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
    """
        For models that inherit from :class:`~transformers.PreTrainedModel`, uses that method to compute the number of
        floating point operations for every backward + forward pass. If using another model, either implement such a
        method in the model or subclass and override this method.

        Args:
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """
    model = self._actual_model(self.model)
    if hasattr(model, 'floating_point_ops'):
        return model.floating_point_ops(inputs)
    else:
        return 0
