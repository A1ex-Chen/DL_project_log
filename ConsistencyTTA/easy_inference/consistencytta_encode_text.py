@torch.no_grad()
def encode_text(self, prompt, max_length=None, padding=True):
    device = self.text_encoder.device
    if max_length is None:
        max_length = self.tokenizer.model_max_length
    batch = self.tokenizer(prompt, max_length=max_length, padding=padding,
        truncation=True, return_tensors='pt')
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    prompt_embeds = self.text_encoder(input_ids=input_ids, attention_mask=
        attention_mask)[0]
    bool_prompt_mask = (attention_mask == 1).to(device)
    return prompt_embeds, bool_prompt_mask
