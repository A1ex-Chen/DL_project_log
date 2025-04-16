@staticmethod
def backward(ctx, grad_y):
    x, s, b, rm, riv, mini_m, mini_riv, bitmask = ctx.saved_variables
    epsilon = ctx.epsilon
    mom = ctx.momentum
    ret_cta = ctx.ret_cta
    my_data = ctx.my_data
    pair_data = ctx.pair_data
    magic = ctx.magic
    pair_data2 = ctx.pair_data2
    pair_data3 = ctx.pair_data3
    bn_group = ctx.bn_group
    bwd_occup = ctx.bwd_occup
    bwd_grid_x = ctx.bwd_grid_x
    multi_stream = ctx.multi_stream
    dx, dz, dscale, dbias = bnp.bn_addrelu_bwd_nhwc(x, grad_y, s, b, rm,
        riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data,
        pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup,
        bwd_grid_x, multi_stream)
    return (dx, dz, dscale, dbias, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None)
