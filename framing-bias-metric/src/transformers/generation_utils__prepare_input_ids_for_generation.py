def _prepare_input_ids_for_generation(self, bos_token_id: int
    ) ->torch.LongTensor:
    if bos_token_id is None:
        raise ValueError(
            '`bos_token_id` has to be defined when no `input_ids` are provided.'
            )
    return torch.ones((1, 1), dtype=torch.long, device=self.device
        ) * bos_token_id
