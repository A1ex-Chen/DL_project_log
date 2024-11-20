def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms,
    prefix=colorstr('TensorFlow Lite:')):
    try:
        import tensorflow as tf
        LOGGER.info(
            f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(im.shape)
        f = str(file).replace('.pt', '-fp16.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            from models.tf import representative_dataset_gen
            dataset = LoadImages(check_dataset(check_yaml(data))['train'],
                img_size=imgsz, auto=False)
            converter.representative_dataset = (lambda :
                representative_dataset_gen(dataset, ncalib=100))
            converter.target_spec.supported_ops = [tf.lite.OpsSet.
                TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.experimental_new_quantizer = True
            f = str(file).replace('.pt', '-int8.tflite')
        if nms or agnostic_nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.
                SELECT_TF_OPS)
        tflite_model = converter.convert()
        open(f, 'wb').write(tflite_model)
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
