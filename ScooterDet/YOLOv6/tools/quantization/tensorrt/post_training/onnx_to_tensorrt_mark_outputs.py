def mark_outputs(network):
    last_layer = network.get_layer(network.num_layers - 1)
    if not last_layer.num_outputs:
        logger.error('Last layer contains no outputs.')
        return
    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))
