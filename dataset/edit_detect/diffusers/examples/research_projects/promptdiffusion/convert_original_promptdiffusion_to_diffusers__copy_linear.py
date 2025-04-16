def _copy_linear(hf_linear, pt_linear):
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias
