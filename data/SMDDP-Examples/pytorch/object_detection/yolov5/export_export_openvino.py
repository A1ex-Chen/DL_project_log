def export_openvino(model, file, half, prefix=colorstr('OpenVINO:')):
    try:
        check_requirements(('openvino-dev',))
        import openvino.inference_engine as ie
        LOGGER.info(
            f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(file).replace('.pt', f'_openvino_model{os.sep}')
        cmd = (
            f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
            )
        subprocess.check_output(cmd.split())
        with open(Path(f) / file.with_suffix('.yaml').name, 'w') as g:
            yaml.dump({'stride': int(max(model.stride)), 'names': model.
                names}, g)
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
