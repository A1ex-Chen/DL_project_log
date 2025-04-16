def _convert_id_to_token(self, index: int) ->Union[bytes, str]:
    """Converts an id to a token, special tokens included"""
    if index in self.decoder:
        return self.decoder[index]
    raise ValueError('unknown ids')
