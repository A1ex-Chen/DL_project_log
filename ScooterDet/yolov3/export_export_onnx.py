@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr(
    'ONNX:')):
    check_requirements('onnx>=1.12.0')
    import onnx
    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')
    output_names = ['output0', 'output1'] if isinstance(model,
        SegmentationModel) else ['output0']
    if dynamic:
        dynamic = {'images': {(0): 'batch', (2): 'height', (3): 'width'}}
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {(0): 'batch', (1): 'anchors'}
            dynamic['output1'] = {(0): 'batch', (2): 'mask_height', (3):
                'mask_width'}
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {(0): 'batch', (1): 'anchors'}
    torch.onnx.export(model.cpu() if dynamic else model, im.cpu() if
        dynamic else im, f, verbose=False, opset_version=opset,
        do_constant_folding=True, input_names=['images'], output_names=
        output_names, dynamic_axes=dynamic or None)
    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else
                'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim
            LOGGER.info(
                f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...'
                )
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx
