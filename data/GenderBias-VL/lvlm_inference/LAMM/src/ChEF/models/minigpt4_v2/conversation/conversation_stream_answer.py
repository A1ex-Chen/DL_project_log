def stream_answer(self, conv, img_list, **kargs):
    generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
    streamer = TextIteratorStreamer(self.model.llama_tokenizer,
        skip_special_tokens=True)
    generation_kwargs['streamer'] = streamer
    thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
    thread.start()
    return streamer
