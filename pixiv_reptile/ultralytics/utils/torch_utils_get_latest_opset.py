def get_latest_opset():
    """Return the second-most recent ONNX opset version supported by this version of PyTorch, adjusted for maturity."""
    if TORCH_1_13:
        return max(int(k[14:]) for k in vars(torch.onnx) if 
            'symbolic_opset' in k) - 1
    version = torch.onnx.producer_version.rsplit('.', 1)[0]
    return {'1.12': 15, '1.11': 14, '1.10': 13, '1.9': 12, '1.8': 12}.get(
        version, 12)
