def stream_producer():
    return self.chat(tokenizer=tokenizer, query=query, streamer=
        ChatStreamer(tokenizer=tokenizer), history=history, max_new_tokens=
        max_new_tokens, do_sample=do_sample, temperature=temperature, top_p
        =top_p, **kwargs)
