@classmethod
def from_pretrained(cls, *args, **kwargs):
    requires_backends(cls, ['transformers', 'torch', 'note_seq'])
