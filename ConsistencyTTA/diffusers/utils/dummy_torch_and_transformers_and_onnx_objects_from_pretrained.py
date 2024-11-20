@classmethod
def from_pretrained(cls, *args, **kwargs):
    requires_backends(cls, ['torch', 'transformers', 'onnx'])
