@classmethod
def from_config(cls, *args, **kwargs):
    requires_backends(cls, ['transformers', 'torch', 'note_seq'])
