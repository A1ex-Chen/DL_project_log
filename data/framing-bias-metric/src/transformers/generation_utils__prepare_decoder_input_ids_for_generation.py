def _prepare_decoder_input_ids_for_generation(self, input_ids: torch.
    LongTensor, decoder_start_token_id: int=None, bos_token_id: int=None,
    **model_kwargs) ->torch.LongTensor:
    if 'decoder_input_ids' in model_kwargs:
        return model_kwargs['decoder_input_ids']
    decoder_start_token_id = self._get_decoder_start_token_id(
        decoder_start_token_id, bos_token_id)
    decoder_input_ids = torch.ones((input_ids.shape[0], 1), dtype=input_ids
        .dtype, device=input_ids.device) * decoder_start_token_id
    return decoder_input_ids
