def append_ca(net_):
    if net_.__class__.__name__ == 'CrossAttention':
        ca_layers.append(net_)
    elif hasattr(net_, 'children'):
        for net__ in net_.children():
            append_ca(net__)
