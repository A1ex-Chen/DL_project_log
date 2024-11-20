def _calc_width(net):
    import numpy as np
    net_params = net.parameters()
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count
