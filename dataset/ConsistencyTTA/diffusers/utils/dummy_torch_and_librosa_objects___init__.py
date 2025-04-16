def __init__(self, *args, **kwargs):
    requires_backends(self, ['torch', 'librosa'])
