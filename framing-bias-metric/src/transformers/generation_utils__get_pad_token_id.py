def _get_pad_token_id(self, pad_token_id: int=None, eos_token_id: int=None
    ) ->int:
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.'
            )
        pad_token_id = eos_token_id
    return pad_token_id
