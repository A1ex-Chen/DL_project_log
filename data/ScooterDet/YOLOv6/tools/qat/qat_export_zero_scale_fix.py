def zero_scale_fix(model, device):
    for k, m in model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or isinstance(m, quant_nn.
            QuantConvTranspose2d):
            weight_amax = m._weight_quantizer._amax.detach().cpu().numpy()
            print(k)
            ones = np.ones_like(weight_amax)
            print('zero scale number = {}'.format(np.sum(weight_amax == 0.0)))
            weight_amax = np.where(weight_amax == 0.0, ones, weight_amax)
            m._weight_quantizer._amax.copy_(torch.from_numpy(weight_amax).
                to(device))
        else:
            continue
