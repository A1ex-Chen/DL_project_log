@try_export
def export_saved_model(self, prefix=colorstr('TensorFlow SavedModel:')):
    """YOLOv8 TensorFlow SavedModel export."""
    cuda = torch.cuda.is_available()
    try:
        import tensorflow as tf
    except ImportError:
        suffix = ('-macos' if MACOS else '-aarch64' if ARM64 else '' if
            cuda else '-cpu')
        version = '>=2.0.0'
        check_requirements(f'tensorflow{suffix}{version}')
        import tensorflow as tf
    check_requirements(('keras', 'tf_keras', 'sng4onnx>=1.0.1',
        'onnx_graphsurgeon>=0.3.26', 'onnx>=1.12.0',
        'onnx2tf>1.17.5,<=1.22.3', 'onnxslim>=0.1.31', 
        'tflite_support<=0.4.3' if IS_JETSON else 'tflite_support',
        'flatbuffers>=23.5.26,<100', 'onnxruntime-gpu' if cuda else
        'onnxruntime'), cmds='--extra-index-url https://pypi.ngc.nvidia.com')
    LOGGER.info(
        f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    check_version(tf.__version__, '>=2.0.0', name='tensorflow', verbose=
        True, msg='https://github.com/ultralytics/ultralytics/issues/5161')
    import onnx2tf
    f = Path(str(self.file).replace(self.file.suffix, '_saved_model'))
    if f.is_dir():
        shutil.rmtree(f)
    onnx2tf_file = Path(
        'calibration_image_sample_data_20x128x128x3_float32.npy')
    if not onnx2tf_file.exists():
        attempt_download_asset(f'{onnx2tf_file}.zip', unzip=True, delete=True)
    self.args.simplify = True
    f_onnx, _ = self.export_onnx()
    np_data = None
    if self.args.int8:
        tmp_file = f / 'tmp_tflite_int8_calibration_images.npy'
        verbosity = 'info'
        if self.args.data:
            f.mkdir()
            images = [batch['img'].permute(0, 2, 3, 1) for batch in self.
                get_int8_calibration_dataloader(prefix)]
            images = torch.cat(images, 0).float()
            np.save(str(tmp_file), images.numpy().astype(np.float32))
            np_data = [['images', tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 
                255]]]]]]
    else:
        verbosity = 'error'
    LOGGER.info(
        f'{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...'
        )
    onnx2tf.convert(input_onnx_file_path=f_onnx, output_folder_path=str(f),
        not_use_onnxsim=True, verbosity=verbosity,
        output_integer_quantized_tflite=self.args.int8, quant_type=
        'per-tensor', custom_input_op_name_np_data_path=np_data)
    yaml_save(f / 'metadata.yaml', self.metadata)
    if self.args.int8:
        tmp_file.unlink(missing_ok=True)
        for file in f.rglob('*_dynamic_range_quant.tflite'):
            file.rename(file.with_name(file.stem.replace(
                '_dynamic_range_quant', '_int8') + file.suffix))
        for file in f.rglob('*_integer_quant_with_int16_act.tflite'):
            file.unlink()
    for file in f.rglob('*.tflite'):
        f.unlink() if 'quant_with_int16_act.tflite' in str(f
            ) else self._add_tflite_metadata(file)
    return str(f), tf.saved_model.load(f, tags=None, options=None)
