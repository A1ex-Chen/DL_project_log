def unk_id(self, proto):
    return proto.trainer_spec.unk_id + self.original_tokenizer.offset
