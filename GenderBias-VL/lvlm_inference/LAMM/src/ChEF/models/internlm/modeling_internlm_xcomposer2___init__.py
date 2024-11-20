def __init__(self, tokenizer) ->None:
    super().__init__()
    self.tokenizer = tokenizer
    self.queue = response_queue
    self.query = query
    self.history = history
    self.response = ''
    self.received_inputs = False
    self.queue.put((self.response, history + [(self.query, self.response)]))
