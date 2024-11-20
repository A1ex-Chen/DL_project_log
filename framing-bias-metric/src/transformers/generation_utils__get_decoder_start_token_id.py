def _get_decoder_start_token_id(self, decoder_start_token_id: int=None,
    bos_token_id: int=None) ->int:
    decoder_start_token_id = (decoder_start_token_id if 
        decoder_start_token_id is not None else self.config.
        decoder_start_token_id)
    bos_token_id = (bos_token_id if bos_token_id is not None else self.
        config.bos_token_id)
    if decoder_start_token_id is not None:
        return decoder_start_token_id
    elif hasattr(self.config, 'decoder') and hasattr(self.config.decoder,
        'decoder_start_token_id'
        ) and self.config.decoder.decoder_start_token_id is not None:
        return self.config.decoder.decoder_start_token_id
    elif bos_token_id is not None:
        return bos_token_id
    elif hasattr(self.config, 'decoder') and hasattr(self.config.decoder,
        'bos_token_id') and self.config.decoder.bos_token_id is not None:
        return self.config.decoder.bos_token_id
    raise ValueError(
        '`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.'
        )
