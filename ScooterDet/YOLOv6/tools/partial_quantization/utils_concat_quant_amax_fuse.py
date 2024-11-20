def concat_quant_amax_fuse(ops_list):
    if len(ops_list) <= 1:
        return
    amax = -1
    for op in ops_list:
        if hasattr(op, '_amax'):
            op_amax = op._amax.detach().item()
        elif hasattr(op, '_input_quantizer'):
            op_amax = op._input_quantizer._amax.detach().item()
        else:
            print('Not quantable op, skip')
            return
        print('op amax = {:7.4f}, amax = {:7.4f}'.format(op_amax, amax))
        if amax < op_amax:
            amax = op_amax
    print('amax = {:7.4f}'.format(amax))
    for op in ops_list:
        if hasattr(op, '_amax'):
            op._amax.fill_(amax)
        elif hasattr(op, '_input_quantizer'):
            op._input_quantizer._amax.fill_(amax)
