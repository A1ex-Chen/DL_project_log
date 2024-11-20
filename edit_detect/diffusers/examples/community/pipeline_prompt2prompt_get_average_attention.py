def get_average_attention(self):
    average_attention = {key: [(item / self.cur_step) for item in self.
        attention_store[key]] for key in self.attention_store}
    return average_attention
