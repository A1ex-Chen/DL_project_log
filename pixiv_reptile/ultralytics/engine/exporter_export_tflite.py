@try_export
def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr(
    'TensorFlow Lite:')):
    """YOLOv8 TensorFlow Lite export."""
    import tensorflow as tf
    LOGGER.info(
        f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    saved_model = Path(str(self.file).replace(self.file.suffix, '_saved_model')
        )
    if self.args.int8:
        f = saved_model / f'{self.file.stem}_int8.tflite'
    elif self.args.half:
        f = saved_model / f'{self.file.stem}_float16.tflite'
    else:
        f = saved_model / f'{self.file.stem}_float32.tflite'
    return str(f), None
