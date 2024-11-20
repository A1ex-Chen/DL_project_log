def export_coreml(model, im, file, int8, half, prefix=colorstr('CoreML:')):
    try:
        check_requirements(('coremltools',))
        import coremltools as ct
        LOGGER.info(
            f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')
        ts = torch.jit.trace(model, im, strict=False)
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.
            shape, scale=1 / 255, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear'
            ) if half else (32, None)
        if bits < 32:
            if platform.system() == 'Darwin':
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=
                        DeprecationWarning)
                    ct_model = (ct.models.neural_network.quantization_utils
                        .quantize_weights(ct_model, bits, mode))
            else:
                print(
                    f'{prefix} quantization only supported on macOS, skipping...'
                    )
        ct_model.save(f)
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ct_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None
