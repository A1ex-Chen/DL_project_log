def run(self):
    hub_version = huggingface_hub.__version__
    pt_version = 'not installed'
    pt_cuda_available = 'NA'
    if is_torch_available():
        import torch
        pt_version = torch.__version__
        pt_cuda_available = torch.cuda.is_available()
    transformers_version = 'not installed'
    if is_transformers_available():
        import transformers
        transformers_version = transformers.__version__
    accelerate_version = 'not installed'
    if is_accelerate_available():
        import accelerate
        accelerate_version = accelerate.__version__
    xformers_version = 'not installed'
    if is_xformers_available():
        import xformers
        xformers_version = xformers.__version__
    info = {'`diffusers` version': version, 'Platform': platform.platform(),
        'Python version': platform.python_version(),
        'PyTorch version (GPU?)': f'{pt_version} ({pt_cuda_available})',
        'Huggingface_hub version': hub_version, 'Transformers version':
        transformers_version, 'Accelerate version': accelerate_version,
        'xFormers version': xformers_version, 'Using GPU in script?':
        '<fill in>', 'Using distributed or parallel set-up in script?':
        '<fill in>'}
    print(
        """
Copy-and-paste the text below in your GitHub issue and FILL OUT the two last points.
"""
        )
    print(self.format_dict(info))
    return info
