def run(self):
    pt_version = 'not installed'
    pt_cuda_available = 'NA'
    if is_torch_available():
        import torch
        pt_version = torch.__version__
        pt_cuda_available = torch.cuda.is_available()
    tf_version = 'not installed'
    tf_cuda_available = 'NA'
    if is_tf_available():
        import tensorflow as tf
        tf_version = tf.__version__
        try:
            tf_cuda_available = tf.test.is_gpu_available()
        except AttributeError:
            tf_cuda_available = bool(tf.config.list_physical_devices('GPU'))
    info = {'`transformers` version': version, 'Platform': platform.
        platform(), 'Python version': platform.python_version(),
        'PyTorch version (GPU?)': '{} ({})'.format(pt_version,
        pt_cuda_available), 'Tensorflow version (GPU?)': '{} ({})'.format(
        tf_version, tf_cuda_available), 'Using GPU in script?': '<fill in>',
        'Using distributed or parallel set-up in script?': '<fill in>'}
    print(
        """
Copy-and-paste the text below in your GitHub issue and FILL OUT the two last points.
"""
        )
    print(self.format_dict(info))
    return info
