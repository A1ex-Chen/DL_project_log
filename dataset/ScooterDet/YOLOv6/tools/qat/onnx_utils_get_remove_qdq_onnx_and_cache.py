def get_remove_qdq_onnx_and_cache(onnx_file):
    model = onnx.load(onnx_file)
    model_wo_qdq, activation_map = onnx_remove_qdqnode(model)
    onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(
        onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model_wo_qdq, os.path.join(onnx_dir, onnx_new_name))
    cache_name = onnx_new_name.replace('.onnx',
        '_add_insert_qdq_calibration.cache')
    save_calib_cache_file(os.path.join(onnx_dir, cache_name), activation_map)
