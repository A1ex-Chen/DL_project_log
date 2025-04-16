@staticmethod
def _update_seq_length_for_generation(sequence_lengths: torch.LongTensor,
    unfinished_sequences: torch.LongTensor, cur_len: int,
    is_eos_in_next_token: torch.BoolTensor) ->Tuple[torch.LongTensor, torch
    .LongTensor]:
    is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()
        ).bool()
    sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len
        )
    unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token)
        .long())
    return sequence_lengths, unfinished_sequences
