def __init__(self, *args, **kwargs):
    requires_backends(self, ['torch', 'transformers', 'k_diffusion'])
