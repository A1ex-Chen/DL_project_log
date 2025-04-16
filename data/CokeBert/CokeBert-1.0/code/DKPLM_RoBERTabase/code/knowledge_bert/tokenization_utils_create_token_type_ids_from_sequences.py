def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return len(token_ids_0) * [0]
    return [0] * len(token_ids_0) + [1] * len(token_ids_1)
