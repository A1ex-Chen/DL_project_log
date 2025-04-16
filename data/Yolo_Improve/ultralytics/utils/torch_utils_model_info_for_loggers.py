def model_info_for_loggers(trainer):
    """
    Return model info dict with useful model information.

    Example:
        YOLOv8n info for loggers
        ```python
        results = {'model/parameters': 3151904,
                   'model/GFLOPs': 8.746,
                   'model/speed_ONNX(ms)': 41.244,
                   'model/speed_TensorRT(ms)': 3.211,
                   'model/speed_PyTorch(ms)': 18.755}
        ```
    """
    if trainer.args.profile:
        from ultralytics.utils.benchmarks import ProfileModels
        results = ProfileModels([trainer.last], device=trainer.device).profile(
            )[0]
        results.pop('model/name')
    else:
        results = {'model/parameters': get_num_params(trainer.model),
            'model/GFLOPs': round(get_flops(trainer.model), 3)}
    results['model/speed_PyTorch(ms)'] = round(trainer.validator.speed[
        'inference'], 3)
    return results
