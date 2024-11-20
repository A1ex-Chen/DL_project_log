def softmax(hidden_state, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(hidden_state.float(), dim=dim)
    else:
        return F.softmax(hidden_state, dim=dim, dtype=torch.float32)
