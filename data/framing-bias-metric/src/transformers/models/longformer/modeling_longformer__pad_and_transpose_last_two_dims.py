@staticmethod
def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
    """pads rows and then flips rows and columns"""
    hidden_states_padded = F.pad(hidden_states_padded, padding)
    hidden_states_padded = hidden_states_padded.view(*hidden_states_padded.
        size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.
        size(-2))
    return hidden_states_padded
