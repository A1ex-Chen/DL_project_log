def __init__(self, model: nn.Module, inputs, inference_func: Optional[
    Callable]=None, allow_non_tensor: bool=False):
    """
        Args:
            model: an nn.Module
            inputs: An input argument or a tuple of input arguments used to call model.
                After flattening, it has to only consist of tensors.
            inference_func: a callable that takes (model, *inputs), calls the
                model with inputs, and return outputs. By default it
                is ``lambda model, *inputs: model(*inputs)``. Can be override
                if you need to call the model differently.
            allow_non_tensor: allow inputs/outputs to contain non-tensor objects.
                This option will filter out non-tensor objects to make the
                model traceable, but ``inputs_schema``/``outputs_schema`` cannot be
                used anymore because inputs/outputs cannot be rebuilt from pure tensors.
                This is useful when you're only interested in the single trace of
                execution (e.g. for flop count), but not interested in
                generalizing the traced graph to new inputs.
        """
    super().__init__()
    if isinstance(model, (nn.parallel.distributed.DistributedDataParallel,
        nn.DataParallel)):
        model = model.module
    self.model = model
    if not isinstance(inputs, tuple):
        inputs = inputs,
    self.inputs = inputs
    self.allow_non_tensor = allow_non_tensor
    if inference_func is None:
        inference_func = lambda model, *inputs: model(*inputs)
    self.inference_func = inference_func
    self.flattened_inputs, self.inputs_schema = flatten_to_tuple(inputs)
    if all(isinstance(x, torch.Tensor) for x in self.flattened_inputs):
        return
    if self.allow_non_tensor:
        self.flattened_inputs = tuple([x for x in self.flattened_inputs if
            isinstance(x, torch.Tensor)])
        self.inputs_schema = None
    else:
        for input in self.flattened_inputs:
            if not isinstance(input, torch.Tensor):
                raise ValueError(
                    f'Inputs for tracing must only contain tensors. Got a {type(input)} instead.'
                    )
