def run(weights=ROOT / 'yolov5s.pt', imgsz=(640, 640), batch_size=1,
    dynamic=False):
    im = torch.zeros((batch_size, 3, *imgsz))
    model = attempt_load(weights, device=torch.device('cpu'), inplace=True,
        fuse=False)
    _ = model(im)
    model.info()
    im = tf.zeros((batch_size, *imgsz, 3))
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    _ = tf_model.predict(im)
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else
        batch_size)
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()
    LOGGER.info(
        """PyTorch, TensorFlow and Keras models successfully verified.
Use export.py for TF model export."""
        )
