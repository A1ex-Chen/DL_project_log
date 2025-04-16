def get_ntk_alpha(self, true_seq_len):
    context_value = math.log(true_seq_len / self.seq_length, 2) + 1
    ntk_alpha = 2 ** math.ceil(context_value) - 1
    ntk_alpha = max(ntk_alpha, 1)
    return ntk_alpha
