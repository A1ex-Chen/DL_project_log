def load(name: str, device: Union[str, torch.device]='cuda' if torch.cuda.
    is_available() else 'cpu', jit: bool=False, download_root: str=None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.
            expanduser('~/.cache/clip'))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f'Model {name} not found; available models = {available_models()}')
    with open(model_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location=device if jit else
                'cpu').eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(
                    f'File {model_path} is not a JIT archive. Loading as a state dict instead'
                    )
                jit = False
            state_dict = torch.load(opened_file, map_location='cpu')
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == 'cpu':
            model.float()
        return model, _transform(model.visual.input_resolution)
    device_holder = torch.jit.trace(lambda : torch.ones([]).to(torch.device
        (device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes(
        'prim::Constant') if 'Device' in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, 'graph') else []
        except RuntimeError:
            graphs = []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']
                    ).startswith('cuda'):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    if str(device) == 'cpu':
        float_holder = torch.jit.trace(lambda : torch.ones([]).float(),
            example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, 'graph') else []
            except RuntimeError:
                graphs = []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model, _transform(model.input_resolution.item())
