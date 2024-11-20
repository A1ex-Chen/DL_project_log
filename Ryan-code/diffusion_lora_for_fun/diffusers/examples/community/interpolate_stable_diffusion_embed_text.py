def embed_text(self, text):
    """takes in text and turns it into text embeddings"""
    text_input = self.tokenizer(text, padding='max_length', max_length=self
        .tokenizer.model_max_length, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
    return embed
