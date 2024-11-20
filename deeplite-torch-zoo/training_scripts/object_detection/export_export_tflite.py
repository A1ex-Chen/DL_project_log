@try_export
def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms,
    prefix=colorstr('TensorFlow Lite:')):
    import tensorflow as tf
    LOGGER.info(
        f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    batch_size, ch, *imgsz = list(im.shape)
    f = str(file) + '-fp16.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:

        def representative_dataset(ncalib=100):
            for _ in range(ncalib):
                data = np.random.rand(1, imgsz[0], imgsz[1], 3)
                yield [data.astype(np.float32)]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.
            TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.experimental_new_quantizer = True
        f = str(file) + '-int8.tflite'
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS
            )
    tflite_model = converter.convert()
    open(f, 'wb').write(tflite_model)
    return f, None
