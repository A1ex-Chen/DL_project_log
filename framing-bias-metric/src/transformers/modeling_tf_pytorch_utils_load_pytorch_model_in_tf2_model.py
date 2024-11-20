def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None,
    allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    pt_state_dict = pt_model.state_dict()
    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict,
        tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys)
