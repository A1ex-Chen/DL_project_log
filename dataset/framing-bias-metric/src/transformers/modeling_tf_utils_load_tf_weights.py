def load_tf_weights(model, resolved_archive_file):
    """
    Detect missing and unexpected layers and load the TF weights accordingly to their names and shapes.

    Args:
        model (:obj:`tf.keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (:obj:`str`):
            The location of the H5 file.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    missing_layers = []
    unexpected_layers = []
    with h5py.File(resolved_archive_file, 'r') as f:
        saved_h5_model_layers_name = set(hdf5_format.
            load_attributes_from_hdf5_group(f, 'layer_names'))
        missing_layers = list(set([layer.name for layer in model.layers]) -
            saved_h5_model_layers_name)
        unexpected_layers = list(saved_h5_model_layers_name - set([layer.
            name for layer in model.layers]))
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []
        for layer in model.layers:
            if layer.name in saved_h5_model_layers_name:
                h5_layer_object = f[layer.name]
                symbolic_weights = (layer.trainable_weights + layer.
                    non_trainable_weights)
                saved_weights = {}
                for weight_name in hdf5_format.load_attributes_from_hdf5_group(
                    h5_layer_object, 'weight_names'):
                    name = '/'.join(weight_name.split('/')[1:])
                    saved_weights[name] = np.asarray(h5_layer_object[
                        weight_name])
                    saved_weight_names_set.add(name)
                for symbolic_weight in symbolic_weights:
                    symbolic_weight_name = '/'.join(symbolic_weight.name.
                        split('/')[1:])
                    saved_weight_value = saved_weights.get(symbolic_weight_name
                        , None)
                    symbolic_weights_names.add(symbolic_weight_name)
                    if saved_weight_value is not None:
                        if K.int_shape(symbolic_weight
                            ) != saved_weight_value.shape:
                            try:
                                array = np.reshape(saved_weight_value, K.
                                    int_shape(symbolic_weight))
                            except AssertionError as e:
                                e.args += K.int_shape(symbolic_weight
                                    ), saved_weight_value.shape
                                raise e
                        else:
                            array = saved_weight_value
                        weight_value_tuples.append((symbolic_weight, array))
    K.batch_set_value(weight_value_tuples)
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set)
        )
    unexpected_layers.extend(list(saved_weight_names_set -
        symbolic_weights_names))
    return missing_layers, unexpected_layers
