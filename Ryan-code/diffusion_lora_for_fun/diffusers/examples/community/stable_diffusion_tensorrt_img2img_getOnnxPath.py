def getOnnxPath(model_name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, model_name + ('.opt' if opt else '') +
        '.onnx')
