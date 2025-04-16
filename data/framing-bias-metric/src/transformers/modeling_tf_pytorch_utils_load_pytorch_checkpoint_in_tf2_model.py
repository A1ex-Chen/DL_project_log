def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path,
    tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error(
            'Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    pt_path = os.path.abspath(pytorch_checkpoint_path)
    logger.info('Loading PyTorch weights from {}'.format(pt_path))
    pt_state_dict = torch.load(pt_path, map_location='cpu')
    logger.info('PyTorch checkpoint contains {:,} parameters'.format(sum(t.
        numel() for t in pt_state_dict.values())))
    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict,
        tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys)
