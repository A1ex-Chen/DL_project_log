def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()
        if m in {'classify', 'classifier', 'cls', 'fc'}:
            return 'classify'
        if 'detect' in m:
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'
        if m == 'obb':
            return 'obb'
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    if isinstance(model, nn.Module):
        for x in ('model.args', 'model.model.args', 'model.model.model.args'):
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in ('model.yaml', 'model.model.yaml', 'model.model.model.yaml'):
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'
            elif isinstance(m, OBB):
                return 'obb'
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return 'detect'
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif '-obb' in model.stem or 'obb' in model.parts:
            return 'obb'
        elif 'detect' in model.parts:
            return 'detect'
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
        )
    return 'detect'
