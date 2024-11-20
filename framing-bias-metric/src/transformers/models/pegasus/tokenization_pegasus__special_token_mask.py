def _special_token_mask(self, seq):
    all_special_ids = set(self.all_special_ids)
    all_special_ids.remove(self.unk_token_id)
    assert all_special_ids == set(range(len(self.additional_special_tokens) +
        3)
        ), f'There should be 3 special tokens: mask_token, pad_token, and eos_token + {len(self.additional_special_tokens)} additional_special_tokens, but got {all_special_ids}'
    return [(1 if x in all_special_ids else 0) for x in seq]
