def export_formats():
    """YOLOv8 export formats."""
    import pandas
    x = [['PyTorch', '-', '.pt', True, True], ['TorchScript', 'torchscript',
        '.torchscript', True, True], ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False], [
        'TensorRT', 'engine', '.engine', False, True], ['CoreML', 'coreml',
        '.mlpackage', True, False], ['TensorFlow SavedModel', 'saved_model',
        '_saved_model', True, True], ['TensorFlow GraphDef', 'pb', '.pb', 
        True, True], ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', True, False],
        ['TensorFlow.js', 'tfjs', '_web_model', True, False], [
        'PaddlePaddle', 'paddle', '_paddle_model', True, True], ['NCNN',
        'ncnn', '_ncnn_model', True, True]]
    return pandas.DataFrame(x, columns=['Format', 'Argument', 'Suffix',
        'CPU', 'GPU'])
