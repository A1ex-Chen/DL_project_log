def __call__(self, *args, **kwargs) ->typing.Union[torch.Tensor, typing.
    Tuple[torch.Tensor, ...]]:
    """ Invokes the model. Parameters can be provided via args or kwargs.
        args, if provided, are assumed to match the order of inputs of the model.
        kwargs are matched with the model input names.
        """
    inputs = self._create_inputs(*args, **kwargs)
    response = self.client.infer(model_name=self.model_name, inputs=inputs)
    result = []
    for output in self.metadata['outputs']:
        tensor = torch.as_tensor(response.as_numpy(output['name']))
        result.append(tensor)
    return result[0] if len(result) == 1 else result
