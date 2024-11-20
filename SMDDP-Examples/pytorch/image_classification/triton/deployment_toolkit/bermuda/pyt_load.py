def load(self, model_path: Union[str, Path], **_) ->Model:
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model = torch.jit.load(model_path.as_posix())
    precision = infer_model_precision(model)
    io_spec = self._io_spec
    if not io_spec:
        yaml_path = model_path.parent / f'{model_path.stem}.yaml'
        if not yaml_path.is_file():
            raise ValueError(
                f'If `--tensor-names-path is not provided, TorchScript model loader expects file {yaml_path} with tensor information.'
                )
        with yaml_path.open('r') as fh:
            tensor_info = yaml.load(fh, Loader=yaml.SafeLoader)
            io_spec = InputOutputSpec(tensor_info['inputs'], tensor_info[
                'outputs'])
    return Model(handle=model, precision=precision, inputs=io_spec.inputs,
        outputs=io_spec.outputs)
