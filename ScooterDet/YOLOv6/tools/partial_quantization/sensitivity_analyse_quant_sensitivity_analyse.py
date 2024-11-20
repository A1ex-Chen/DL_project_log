def quant_sensitivity_analyse(model_ptq, evaler):
    model_quant_disable(model_ptq)
    quant_sensitivity = list()
    for k, m in model_ptq.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or isinstance(m, quant_nn.
            QuantConvTranspose2d) or isinstance(m, quant_nn.MaxPool2d):
            module_quant_enable(model_ptq, k)
        else:
            continue
        eval_result = evaler.eval(model_ptq)
        print(eval_result)
        print(
            'Quantize Layer {}, result mAP0.5 = {:0.4f}, mAP0.5:0.95 = {:0.4f}'
            .format(k, eval_result[0], eval_result[1]))
        quant_sensitivity.append((k, eval_result[0], eval_result[1]))
        module_quant_disable(model_ptq, k)
    return quant_sensitivity
