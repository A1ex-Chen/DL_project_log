def encode_text(self, text, add_special_tokens=False):
    token = self.tokenizer(text, return_tensors='pt', add_special_tokens=
        add_special_tokens).input_ids.to(self.device)
    embs = self.model.tok_embeddings(token)
    return embs
