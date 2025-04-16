def _get_output_for_vectorized_inputs(self, hidden_states):
    hidden_states = self.norm_out(hidden_states)
    logits = self.out(hidden_states)
    logits = logits.permute(0, 2, 1)
    output = F.log_softmax(logits.double(), dim=1).float()
    return output
