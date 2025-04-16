def partial_quant(model_ptq, quantable_ops=None):
    for k, m in model_ptq.named_modules():
        if quantable_op_check(k, quantable_ops):
            continue
        if isinstance(m, quant_nn.QuantConv2d) or isinstance(m, quant_nn.
            QuantConvTranspose2d) or isinstance(m, quant_nn.QuantMaxPool2d):
            module_quant_disable(model_ptq, k)
