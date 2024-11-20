@staticmethod
def generate_results_dict(model_name, t_onnx, t_engine, model_info):
    """Generates a dictionary of model details including name, parameters, GFLOPS and speed metrics."""
    layers, params, gradients, flops = model_info
    return {'model/name': model_name, 'model/parameters': params,
        'model/GFLOPs': round(flops, 3), 'model/speed_ONNX(ms)': round(
        t_onnx[0], 3), 'model/speed_TensorRT(ms)': round(t_engine[0], 3)}
