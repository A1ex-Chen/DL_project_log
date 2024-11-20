def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=
    colorstr('ONNX:')):
    try:
        check_requirements(('onnx',))
        import onnx
        LOGGER.info(
            f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        torch.onnx.export(model.cpu() if dynamic else model, im.cpu() if
            dynamic else im, f, verbose=False, opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.
            onnx.TrainingMode.EVAL, do_constant_folding=not train,
            input_names=['images'], output_names=['output'], dynamic_axes={
            'images': {(0): 'batch', (2): 'height', (3): 'width'}, 'output':
            {(0): 'batch', (1): 'anchors'}} if dynamic else None)
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
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')
