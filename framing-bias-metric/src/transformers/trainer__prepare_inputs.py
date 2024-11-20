def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) ->Dict[
    str, Union[torch.Tensor, Any]]:
    """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(self.args.device)
    if self.args.past_index >= 0 and self._past is not None:
        inputs['mems'] = self._past
    return inputs
