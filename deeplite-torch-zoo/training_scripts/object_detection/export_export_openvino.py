@try_export
def export_openvino(file, metadata, half, prefix=colorstr('OpenVINO:')):
    check_requirements('openvino-dev')
    import openvino.inference_engine as ie
    LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...'
        )
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')
    args = ['mo', '--input_model', str(file.with_suffix('.onnx')),
        '--output_dir', f, '--data_type', 'FP16' if half else 'FP32']
    subprocess.run(args, check=True, env=os.environ)
    return f, None
