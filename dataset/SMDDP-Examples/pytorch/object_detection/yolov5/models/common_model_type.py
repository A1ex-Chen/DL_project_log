@staticmethod
def model_type(p='path/to/model.pt'):
    from export import export_formats
    suffixes = list(export_formats().Suffix) + ['.xml']
    check_suffix(p, suffixes)
    p = Path(p).name
    (pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu,
        tfjs, xml2) = (s in p for s in suffixes)
    xml |= xml2
    tflite &= not edgetpu
    return (pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite,
        edgetpu, tfjs)
