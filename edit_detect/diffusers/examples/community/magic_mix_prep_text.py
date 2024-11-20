def prep_text(self, prompt):
    text_input = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]
    uncond_input = self.tokenizer('', padding='max_length', max_length=self
        .tokenizer.model_max_length, truncation=True, return_tensors='pt')
    uncond_embedding = self.text_encoder(uncond_input.input_ids.to(self.device)
        )[0]
    return torch.cat([uncond_embedding, text_embedding])
