@try_export
def export_tfjs(self, prefix=colorstr('TensorFlow.js:')):
    """YOLOv8 TensorFlow.js export."""
    check_requirements('tensorflowjs')
    if ARM64:
        check_requirements('numpy==1.23.5')
    import tensorflow as tf
    import tensorflowjs as tfjs
    LOGGER.info(
        f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
    f = str(self.file).replace(self.file.suffix, '_web_model')
    f_pb = str(self.file.with_suffix('.pb'))
    gd = tf.Graph().as_graph_def()
    with open(f_pb, 'rb') as file:
        gd.ParseFromString(file.read())
    outputs = ','.join(gd_outputs(gd))
    LOGGER.info(f'\n{prefix} output node names: {outputs}')
    quantization = ('--quantize_float16' if self.args.half else 
        '--quantize_uint8' if self.args.int8 else '')
    with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:
        cmd = (
            f'tensorflowjs_converter --input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
            )
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
    if ' ' in f:
        LOGGER.warning(
            f"{prefix} WARNING ⚠️ your model may not work correctly with spaces in path '{f}'."
            )
    yaml_save(Path(f) / 'metadata.yaml', self.metadata)
    return f, None
