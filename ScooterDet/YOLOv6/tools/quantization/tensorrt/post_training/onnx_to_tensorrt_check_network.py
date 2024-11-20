def check_network(network):
    if not network.num_outputs:
        logger.warning(
            "No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong."
            )
        mark_outputs(network)
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for
        out in outputs])
    logger.debug('=== Network Description ===')
    for i, inp in enumerate(inputs):
        logger.debug('Input  {0} | Name: {1:{2}} | Shape: {3}'.format(i,
            inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.debug('Output {0} | Name: {1:{2}} | Shape: {3}'.format(i,
            out.name, max_len, out.shape))
