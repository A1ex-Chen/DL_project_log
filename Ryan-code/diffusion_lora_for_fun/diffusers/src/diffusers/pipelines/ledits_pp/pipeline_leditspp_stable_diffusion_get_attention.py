def get_attention(self, step: int):
    if self.average:
        attention = {key: [(item / self.cur_step) for item in self.
            attention_store[key]] for key in self.attention_store}
    else:
        assert step is not None
        attention = self.attention_store[step]
    return attention
