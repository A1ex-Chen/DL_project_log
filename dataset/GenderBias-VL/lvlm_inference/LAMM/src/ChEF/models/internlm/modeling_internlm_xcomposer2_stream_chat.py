@torch.no_grad()
def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]]
    =[], max_new_tokens: int=1024, do_sample: bool=True, temperature: float
    =0.8, top_p: float=0.8, **kwargs):
    """Return a generator in format: (response, history) Eg.

        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')]) ('你好，有什么可以帮助您的吗？', [('你好',
        '你好，有什么可以帮助您的吗？')])
        """
    if BaseStreamer is None:
        raise ModuleNotFoundError(
            'The version of `transformers` is too low. Please make sure that you have installed `transformers>=4.28.0`.'
            )
    response_queue = queue.Queue(maxsize=20)


    class ChatStreamer(BaseStreamer):

        def __init__(self, tokenizer) ->None:
            super().__init__()
            self.tokenizer = tokenizer
            self.queue = response_queue
            self.query = query
            self.history = history
            self.response = ''
            self.received_inputs = False
            self.queue.put((self.response, history + [(self.query, self.
                response)]))

        def put(self, value):
            if len(value.shape) > 1 and value.shape[0] > 1:
                raise ValueError('ChatStreamer only supports batch size 1')
            elif len(value.shape) > 1:
                value = value[0]
            if not self.received_inputs:
                self.received_inputs = True
                return
            token = self.tokenizer.decode([value[-1]], skip_special_tokens=True
                )
            if token.strip() != '[UNUSED_TOKEN_145]':
                self.response = self.response + token
                history = self.history + [(self.query, self.response)]
                self.queue.put((self.response, history))

        def end(self):
            self.queue.put(None)

    def stream_producer():
        return self.chat(tokenizer=tokenizer, query=query, streamer=
            ChatStreamer(tokenizer=tokenizer), history=history,
            max_new_tokens=max_new_tokens, do_sample=do_sample, temperature
            =temperature, top_p=top_p, **kwargs)

    def consumer():
        producer = threading.Thread(target=stream_producer)
        producer.start()
        while True:
            res = response_queue.get()
            if res is None:
                return
            yield res
    return consumer()
