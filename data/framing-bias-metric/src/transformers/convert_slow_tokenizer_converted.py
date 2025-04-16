def converted(self) ->Tokenizer:
    tokenizer = self.tokenizer(self.proto)
    tokenizer.normalizer = self.normalizer(self.proto)
    replacement = '▁'
    add_prefix_space = True
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.
        WhitespaceSplit(), pre_tokenizers.Metaspace(replacement=replacement,
        add_prefix_space=add_prefix_space)])
    tokenizer.decoder = decoders.Metaspace(replacement=replacement,
        add_prefix_space=add_prefix_space)
    post_processor = self.post_processor()
    if post_processor:
        tokenizer.post_processor = post_processor
    return tokenizer
