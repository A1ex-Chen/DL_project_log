@classmethod
def from_config(cls, *args, **kwargs):
    requires_backends(cls, ['note_seq'])
