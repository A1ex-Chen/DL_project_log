def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=
    False):
    """ Load TF 2.0 model in a pytorch model
    """
    weights = tf_model.weights
    return load_tf2_weights_in_pytorch_model(pt_model, weights,
        allow_missing_keys=allow_missing_keys)
