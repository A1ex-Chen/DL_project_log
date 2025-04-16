def __repr__(self) ->str:
    return (
        f"{'PreTrainedTokenizerFast' if self.is_fast else 'PreTrainedTokenizer'}(name_or_path='{self.name_or_path}', vocab_size={self.vocab_size}, model_max_len={self.model_max_length}, is_fast={self.is_fast}, padding_side='{self.padding_side}', special_tokens={self.special_tokens_map_extended})"
        )
