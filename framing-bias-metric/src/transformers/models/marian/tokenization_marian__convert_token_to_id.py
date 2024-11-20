def _convert_token_to_id(self, token):
    return self.encoder.get(token, self.encoder[self.unk_token])
