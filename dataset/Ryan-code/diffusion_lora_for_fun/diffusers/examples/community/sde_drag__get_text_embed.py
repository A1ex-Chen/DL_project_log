@torch.no_grad()
def _get_text_embed(self, prompt):
    text_input = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0
        ]
    return text_embeddings
