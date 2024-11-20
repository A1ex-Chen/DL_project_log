@staticmethod
def _mask_invalid_locations(input_tensor, affected_seq_len) ->torch.Tensor:
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, 
        affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]
    ending_mask = beginning_mask.flip(dims=(1, 3))
    beginning_input = input_tensor[:, :affected_seq_len, :, :
        affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask == 1, -float('inf'))
    ending_input = input_tensor[:, -affected_seq_len:, :, -(
        affected_seq_len + 1):]
    ending_mask = ending_mask.expand(ending_input.size())
    ending_input.masked_fill_(ending_mask == 1, -float('inf'))
