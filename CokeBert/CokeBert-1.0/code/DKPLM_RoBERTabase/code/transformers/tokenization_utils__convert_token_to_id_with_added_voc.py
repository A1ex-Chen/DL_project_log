def _convert_token_to_id_with_added_voc(self, token):
    if token is None:
        return None
    if token in self.added_tokens_encoder:
        return self.added_tokens_encoder[token]
    return self._convert_token_to_id(token)
