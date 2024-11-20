def _add_eos_if_not_present(self, token_ids: List[int]) ->List[int]:
    """Do not add eos again if user already added it."""
    if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
        warnings.warn(
            f'This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added.'
            )
        return token_ids
    else:
        return token_ids + [self.eos_token_id]
