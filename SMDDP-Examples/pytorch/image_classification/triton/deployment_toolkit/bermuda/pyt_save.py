def save(self, model: Model, model_path: Union[str, Path]) ->None:
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    if isinstance(model.handle, torch.jit.ScriptModule):
        torch.jit.save(model.handle, model_path.as_posix())
    else:
        print(
            "The model must be of type 'torch.jit.ScriptModule'. Saving aborted."
            )
        assert False

    def _format_tensor_spec(tensor_spec):
        tensor_spec = tensor_spec._replace(shape=list(tensor_spec.shape))
        tensor_spec = dict(tensor_spec._asdict())
        return tensor_spec
    tensor_specs = {'inputs': {k: _format_tensor_spec(v) for k, v in model.
        inputs.items()}, 'outputs': {k: _format_tensor_spec(v) for k, v in
        model.outputs.items()}}
    yaml_path = model_path.parent / f'{model_path.stem}.yaml'
    with Path(yaml_path).open('w') as fh:
        yaml.dump(tensor_specs, fh, indent=4)
