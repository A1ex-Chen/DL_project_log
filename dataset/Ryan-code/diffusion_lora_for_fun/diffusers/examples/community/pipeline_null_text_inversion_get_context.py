def get_context(self, prompt):
    uncond_input = self.tokenizer([''], padding='max_length', max_length=
        self.tokenizer.model_max_length, return_tensors='pt')
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.
        device))[0]
    text_input = self.tokenizer([prompt], padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0
        ]
    context = torch.cat([uncond_embeddings, text_embeddings])
    return context
