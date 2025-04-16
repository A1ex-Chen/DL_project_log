def quant_model_init(model, device):
    model_ptq = copy.deepcopy(model)
    model_ptq.eval()
    model_ptq.to(device)
    conv2d_weight_default_desc = (tensor_quant.
        QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
    conv2d_input_default_desc = QuantDescriptor(num_bits=8, calib_method=
        'histogram')
    convtrans2d_weight_default_desc = (tensor_quant.
        QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL)
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=8,
        calib_method='histogram')
    for k, m in model_ptq.named_modules():
        if 'proj_conv' in k:
            print('Skip Layer {}'.format(k))
            continue
        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels, out_channels,
                kernel_size, stride, padding, quant_desc_input=
                conv2d_input_default_desc, quant_desc_weight=
                conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model_ptq, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                out_channels, kernel_size, stride, padding,
                quant_desc_input=convtrans2d_input_default_desc,
                quant_desc_weight=convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model_ptq, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size, stride,
                padding, dilation, ceil_mode, quant_desc_input=
                conv2d_input_default_desc)
            set_module(model_ptq, k, quant_maxpool2d)
        else:
            continue
    return model_ptq.to(device)
