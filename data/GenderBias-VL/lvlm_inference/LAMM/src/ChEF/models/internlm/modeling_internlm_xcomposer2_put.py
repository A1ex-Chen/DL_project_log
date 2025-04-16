def put(self, value):
    if len(value.shape) > 1 and value.shape[0] > 1:
        raise ValueError('ChatStreamer only supports batch size 1')
    elif len(value.shape) > 1:
        value = value[0]
    if not self.received_inputs:
        self.received_inputs = True
        return
    token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
    if token.strip() != '[UNUSED_TOKEN_145]':
        self.response = self.response + token
        history = self.history + [(self.query, self.response)]
        self.queue.put((self.response, history))
