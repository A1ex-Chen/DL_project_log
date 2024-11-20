def step_callback(self, x_t):
    if self.local_blend is not None:
        x_t = self.local_blend(x_t, self.attention_store)
    return x_t
