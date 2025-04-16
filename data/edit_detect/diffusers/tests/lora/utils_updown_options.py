def updown_options(blocks_with_tf, layers_per_block, value):
    """
            Generate every possible combination for how a lora weight dict for the up/down part can be.
            E.g. 2, {"block_1": 2}, {"block_1": [2,2,2]}, {"block_1": 2, "block_2": [2,2,2]}, ...
            """
    num_val = value
    list_val = [value] * layers_per_block
    node_opts = [None, num_val, list_val]
    node_opts_foreach_block = [node_opts] * len(blocks_with_tf)
    updown_opts = [num_val]
    for nodes in product(*node_opts_foreach_block):
        if all(n is None for n in nodes):
            continue
        opt = {}
        for b, n in zip(blocks_with_tf, nodes):
            if n is not None:
                opt['block_' + str(b)] = n
        updown_opts.append(opt)
    return updown_opts
