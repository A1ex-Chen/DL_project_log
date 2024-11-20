def prompt_wrap(self, img_embeds, prompt):
    batch_size = img_embeds.shape[0]
    p_before, p_after = prompt.split('<ImageHere>')
    p_before_tokens = self.tokenizer(p_before, return_tensors='pt',
        add_special_tokens=True).to(img_embeds.device)
    p_before_embeds = self.model.tok_embeddings(p_before_tokens.input_ids
        ).expand(batch_size, -1, -1)
    wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds], dim=1)
    wrapped_atts_img = torch.ones(wrapped_img_embeds.size()[:-1], dtype=
        torch.long).to(img_embeds.device)
    wrapped_target = torch.ones(batch_size, wrapped_img_embeds.shape[1],
        dtype=torch.long).to(img_embeds.device) * -100
    return wrapped_img_embeds, wrapped_atts_img, wrapped_target
