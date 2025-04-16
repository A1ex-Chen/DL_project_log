@torch.no_grad()
def get_embeds(self, prompt: List[str], batch_size: int=16
    ) ->torch.FloatTensor:
    num_prompts = len(prompt)
    embeds = []
    for i in range(0, num_prompts, batch_size):
        prompt_slice = prompt[i:i + batch_size]
        input_ids = self.tokenizer(prompt_slice, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt').input_ids
        input_ids = input_ids.to(self.text_encoder.device)
        embeds.append(self.text_encoder(input_ids)[0])
    return torch.cat(embeds, dim=0).mean(0)[None]
