@try_export
def export_saved_model(model, im, file, dynamic, tf_nms=False, agnostic_nms
    =False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=
    0.25, keras=False, prefix=colorstr('TensorFlow SavedModel:')):
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(
            f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}"
            )
        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from deeplite_torch_zoo.src.object_detection.yolo.tf import TFModel
    LOGGER.info(
        f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im.shape)
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all,
        iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else
        batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class,
        topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format='tf')
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.
            inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else
            frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(tfm, f, options=tf.saved_model.SaveOptions(
            experimental_custom_gradients=False) if check_version(tf.
            __version__, '2.6') else tf.saved_model.SaveOptions())
    return f, keras_model
