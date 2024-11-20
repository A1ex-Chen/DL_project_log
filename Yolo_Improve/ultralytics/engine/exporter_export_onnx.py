@try_export
def export_onnx(self, prefix=colorstr('ONNX:')):
    """YOLOv8 ONNX export."""
    requirements = ['onnx>=1.12.0']
    if self.args.simplify:
        requirements += ['onnxslim>=0.1.31', 'onnxruntime' + ('-gpu' if
            torch.cuda.is_available() else '')]
    check_requirements(requirements)
    import onnx
    opset_version = self.args.opset or get_latest_opset()
    LOGGER.info(
        f'\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...'
        )
    f = str(self.file.with_suffix('.onnx'))
    output_names = ['output0', 'output1'] if isinstance(self.model,
        SegmentationModel) else ['output0']
    dynamic = self.args.dynamic
    if dynamic:
        dynamic = {'images': {(0): 'batch', (2): 'height', (3): 'width'}}
        if isinstance(self.model, SegmentationModel):
            dynamic['output0'] = {(0): 'batch', (2): 'anchors'}
            dynamic['output1'] = {(0): 'batch', (2): 'mask_height', (3):
                'mask_width'}
        elif isinstance(self.model, DetectionModel):
            dynamic['output0'] = {(0): 'batch', (2): 'anchors'}
    torch.onnx.export(self.model.cpu() if dynamic else self.model, self.im.
        cpu() if dynamic else self.im, f, verbose=False, opset_version=
        opset_version, do_constant_folding=True, input_names=['images'],
        output_names=output_names, dynamic_axes=dynamic or None)
    model_onnx = onnx.load(f)
    if self.args.simplify:
        try:
            import onnxslim
            LOGGER.info(
                f'{prefix} slimming with onnxslim {onnxslim.__version__}...')
            model_onnx = onnxslim.slim(model_onnx)
        except Exception as e:
            LOGGER.warning(f'{prefix} simplifier failure: {e}')
    for k, v in self.metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)
    return f, model_onnx
