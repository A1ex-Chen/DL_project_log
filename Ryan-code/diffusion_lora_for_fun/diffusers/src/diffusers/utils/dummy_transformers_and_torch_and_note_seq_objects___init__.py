def __init__(self, *args, **kwargs):
    requires_backends(self, ['transformers', 'torch', 'note_seq'])
