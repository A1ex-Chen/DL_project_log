@try_export
def export_coreml(self, prefix=colorstr('CoreML:')):
    """YOLOv8 CoreML export."""
    mlmodel = self.args.format.lower() == 'mlmodel'
    check_requirements('coremltools>=6.0,<=6.2' if mlmodel else
        'coremltools>=7.0')
    import coremltools as ct
    LOGGER.info(
        f'\n{prefix} starting export with coremltools {ct.__version__}...')
    assert not WINDOWS, 'CoreML export is not supported on Windows, please run on macOS or Linux.'
    assert self.args.batch == 1, "CoreML batch sizes > 1 are not supported. Please retry at 'batch=1'."
    f = self.file.with_suffix('.mlmodel' if mlmodel else '.mlpackage')
    if f.is_dir():
        shutil.rmtree(f)
    bias = [0.0, 0.0, 0.0]
    scale = 1 / 255
    classifier_config = None
    if self.model.task == 'classify':
        classifier_config = ct.ClassifierConfig(list(self.model.names.values())
            ) if self.args.nms else None
        model = self.model
    elif self.model.task == 'detect':
        model = IOSDetectModel(self.model, self.im
            ) if self.args.nms else self.model
    else:
        if self.args.nms:
            LOGGER.warning(
                f"{prefix} WARNING ⚠️ 'nms=True' is only available for Detect models like 'yolov8n.pt'."
                )
        model = self.model
    ts = torch.jit.trace(model.eval(), self.im, strict=False)
    ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=self.im.
        shape, scale=scale, bias=bias)], classifier_config=
        classifier_config, convert_to='neuralnetwork' if mlmodel else
        'mlprogram')
    bits, mode = (8, 'kmeans') if self.args.int8 else (16, 'linear'
        ) if self.args.half else (32, None)
    if bits < 32:
        if 'kmeans' in mode:
            check_requirements('scikit-learn')
        if mlmodel:
            ct_model = (ct.models.neural_network.quantization_utils.
                quantize_weights(ct_model, bits, mode))
        elif bits == 8:
            import coremltools.optimize.coreml as cto
            op_config = cto.OpPalettizerConfig(mode='kmeans', nbits=bits,
                weight_threshold=512)
            config = cto.OptimizationConfig(global_config=op_config)
            ct_model = cto.palettize_weights(ct_model, config=config)
    if self.args.nms and self.model.task == 'detect':
        if mlmodel:
            check_version(PYTHON_VERSION, '<3.11', name='Python ', hard=True)
            weights_dir = None
        else:
            ct_model.save(str(f))
            weights_dir = str(f / 'Data/com.apple.CoreML/weights')
        ct_model = self._pipeline_coreml(ct_model, weights_dir=weights_dir)
    m = self.metadata
    ct_model.short_description = m.pop('description')
    ct_model.author = m.pop('author')
    ct_model.license = m.pop('license')
    ct_model.version = m.pop('version')
    ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
    try:
        ct_model.save(str(f))
    except Exception as e:
        LOGGER.warning(
            f'{prefix} WARNING ⚠️ CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928.'
            )
        f = f.with_suffix('.mlmodel')
        ct_model.save(str(f))
    return f, ct_model
