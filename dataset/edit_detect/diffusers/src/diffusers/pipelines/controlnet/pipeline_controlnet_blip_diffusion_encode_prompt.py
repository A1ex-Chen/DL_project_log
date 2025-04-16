def encode_prompt(self, query_embeds, prompt, device=None):
    device = device or self._execution_device
    max_len = self.text_encoder.text_model.config.max_position_embeddings
    max_len -= self.qformer.config.num_query_tokens
    tokenized_prompt = self.tokenizer(prompt, padding='max_length',
        truncation=True, max_length=max_len, return_tensors='pt').to(device)
    batch_size = query_embeds.shape[0]
    ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size
    text_embeddings = self.text_encoder(input_ids=tokenized_prompt.
        input_ids, ctx_embeddings=query_embeds, ctx_begin_pos=ctx_begin_pos)[0]
    return text_embeddings
