def tokenize_prompt(self, tokenizer):
    """Tokenizes the prompt for this diffusion region using a given tokenizer"""
    self.tokenized_prompt = tokenizer(self.prompt, padding='max_length',
        max_length=tokenizer.model_max_length, truncation=True,
        return_tensors='pt')
