@try_export
def export_coreml(model, im, file, int8, half, prefix=colorstr('CoreML:')):
    check_requirements('coremltools')
    import coremltools as ct
    LOGGER.info(
        f'\n{prefix} starting export with coremltools {ct.__version__}...')
    f = file.with_suffix('.mlmodel')
    ts = torch.jit.trace(model, im, strict=False)
    ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape,
        scale=1 / 255, bias=[0, 0, 0])])
    bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (
        32, None)
    if bits < 32:
        if MACOS:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                ct_model = (ct.models.neural_network.quantization_utils.
                    quantize_weights(ct_model, bits, mode))
        else:
            LOGGER.info(
                f'{prefix} quantization only supported on macOS, skipping...')
    ct_model.save(f)
    return f, ct_model
