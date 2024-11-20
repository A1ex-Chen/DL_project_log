@staticmethod
def _init_sequence_length_for_generation(input_ids: torch.LongTensor,
    max_length: int) ->Tuple[torch.Tensor, torch.Tensor, int]:
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)
    cur_len = input_ids.shape[-1]
    return sequence_lengths, unfinished_sequences, cur_len
