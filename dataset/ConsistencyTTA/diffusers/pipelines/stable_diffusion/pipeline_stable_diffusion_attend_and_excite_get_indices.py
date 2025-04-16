def get_indices(self, prompt: str) ->Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alte"""
    ids = self.tokenizer(prompt).input_ids
    indices = {i: tok for tok, i in zip(self.tokenizer.
        convert_ids_to_tokens(ids), range(len(ids)))}
    return indices
