def make_emblist(self, prompts):
    with torch.no_grad():
        tokens = self.tokenizer(prompts, max_length=self.tokenizer.
            model_max_length, padding=True, truncation=True, return_tensors
            ='pt').input_ids.to(self.device)
        embs = self.text_encoder(tokens, output_hidden_states=True
            ).last_hidden_state.to(self.device, dtype=self.dtype)
    return embs
