def prompt_wrap(self, img_embeds, atts_img, prompt):
    if prompt:
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors=
            'pt', add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(p_after, return_tensors='pt',
            add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens
            .input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens
            .input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds,
            p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.
            shape[1])
        return wrapped_img_embeds, wrapped_atts_img
    else:
        return img_embeds, atts_img
