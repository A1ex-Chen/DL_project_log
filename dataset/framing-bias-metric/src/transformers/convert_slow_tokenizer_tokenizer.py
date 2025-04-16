def tokenizer(self, proto):
    model_type = proto.trainer_spec.model_type
    vocab = self.vocab(proto)
    unk_id = self.unk_id(proto)
    if model_type == 1:
        tokenizer = Tokenizer(Unigram(vocab, unk_id))
    elif model_type == 2:
        vocab, merges = SentencePieceExtractor(self.original_tokenizer.
            vocab_file).extract()
        tokenizer = Tokenizer(BPE(vocab, merges, unk_token=proto.
            trainer_spec.unk_piece, fuse_unk=True))
    else:
        raise Exception(
            "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )
    return tokenizer
