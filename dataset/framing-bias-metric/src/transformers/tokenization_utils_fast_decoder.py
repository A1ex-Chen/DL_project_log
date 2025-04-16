@property
def decoder(self) ->DecoderFast:
    """
        :obj:`tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
    return self._tokenizer._tokenizer.decoder
