def interleav_wrap_chat(self, tokenizer, prompt, image):
    im_len = image.shape[1]
    image_nums = len(image)
    parts = prompt.split('<ImageHere>')
    wrap_embeds, wrap_im_mask = [], []
    temp_len = 0
    for idx, part in enumerate(parts):
        if len(part) > 0:
            part_tokens = tokenizer(part, return_tensors='pt').to(self.device)
            part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
            wrap_embeds.append(part_embeds)
            wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
            temp_len += part_embeds.shape[1]
        if idx < image_nums:
            wrap_embeds.append(image[idx].unsqueeze(0))
            wrap_im_mask.append(torch.ones(1, image[idx].shape[0]))
            temp_len += im_len
        if temp_len > self.max_length:
            break
    wrap_embeds = torch.cat(wrap_embeds, dim=1)
    wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
    wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
    wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device).bool()
    inputs = {'inputs_embeds': wrap_embeds}
    return inputs, wrap_im_mask
