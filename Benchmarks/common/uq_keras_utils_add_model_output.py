def add_model_output(modelIn, mode=None, num_add=None, activation=None):
    """This function modifies the last dense layer in the passed keras model. The modification includes adding units and optionally changing the activation function.

    Parameters
    ----------
    modelIn : keras model
        Keras model to be modified.
    mode : string
        Mode to modify the layer. It could be:
        'abstain' for adding an arbitrary number of units for the abstention optimization strategy.
        'qtl' for quantile regression which needs the outputs to be tripled.
        'het' for heteroscedastic regression which needs the outputs to be doubled.
    num_add : integer
        Number of units to add. This only applies to the 'abstain' mode.
    activation : string
        String with keras specification of activation function (e.g. 'relu', 'sigomid', 'softmax', etc.)

    Return
    ----------
    modelOut : keras model
        Keras model after last dense layer has been modified as specified. If there is no mode specified it returns the same model. If the mode is not one of 'abstain', 'qtl' or 'het' an exception is raised.
    """
    if mode is None:
        return modelIn
    numlayers = len(modelIn.layers)
    i = -1
    while 'dense' not in modelIn.layers[i].name and i + numlayers > 0:
        i -= 1
    assert i + numlayers >= 0
    assert 'dense' in modelIn.layers[i].name
    if mode == 'abstain':
        assert num_add is not None
        new_output_size = modelIn.layers[i].output_shape[-1] + num_add
    elif mode == 'qtl':
        new_output_size = 3 * modelIn.layers[i].output_shape[-1]
    elif mode == 'het':
        new_output_size = 2 * modelIn.layers[i].output_shape[-1]
    else:
        raise Exception(
            'ERROR ! Type of mode specified for adding outputs to the model: '
             + mode + ' not implemented... Exiting')
    config = modelIn.layers[i].get_config()
    config['units'] = new_output_size
    if activation is not None:
        config['activation'] = activation
    if mode == 'het' or mode == 'qtl':
        config['bias_initializer'] = 'ones'
    reconstructed_layer = Dense.from_config(config)
    additional = reconstructed_layer(modelIn.layers[i - 1].output)
    if i < -1:
        for j in range(i + 1, 0):
            config_j = modelIn.layers[j].get_config()
            aux_j = layers.deserialize({'class_name': modelIn.layers[j].
                __class__.__name__, 'config': config_j})
            reconstructed_layer = aux_j.from_config(config_j)
            additional = reconstructed_layer(additional)
    modelOut = Model(modelIn.input, additional)
    return modelOut
