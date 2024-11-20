def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
    """Generates a formatted string for a table row that includes model performance and metric details."""
    layers, params, gradients, flops = model_info
    return (
        f'| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.2f} ± {t_onnx[1]:.2f} ms | {t_engine[0]:.2f} ± {t_engine[1]:.2f} ms | {params / 1000000.0:.1f} | {flops:.1f} |'
        )
