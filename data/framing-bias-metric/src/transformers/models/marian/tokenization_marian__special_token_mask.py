def _special_token_mask(self, seq):
    all_special_ids = set(self.all_special_ids)
    all_special_ids.remove(self.unk_token_id)
    return [(1 if x in all_special_ids else 0) for x in seq]
