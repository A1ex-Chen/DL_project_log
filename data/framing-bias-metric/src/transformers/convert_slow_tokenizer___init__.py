def __init__(self, *args):
    requires_protobuf(self)
    super().__init__(*args)
    from .utils import sentencepiece_model_pb2 as model_pb2
    m = model_pb2.ModelProto()
    m.ParseFromString(open(self.original_tokenizer.vocab_file, 'rb').read())
    self.proto = m
