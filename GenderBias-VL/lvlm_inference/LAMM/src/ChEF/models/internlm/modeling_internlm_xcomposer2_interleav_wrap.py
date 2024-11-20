def interleav_wrap(self, img_list, text_list):
    wrap_embeds_list, wrap_atts_list = [], []
    wrap_target_list, wrap_im_mask_list = [], []
    for image, text in zip(img_list, text_list):
        img_embeds, atts_img, img_target = self.img2emb(image)
        text = text[0]
        parts = text.split('<ImageHere>')
        wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
        temp_len = 0
        image_nums, im_len = img_embeds.shape[:2]
        need_bos = True
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.tokenizer(part, return_tensors='pt',
                    padding='longest', add_special_tokens=need_bos).to(self
                    .device)
                if need_bos:
                    need_bos = False
                wrap_tokens.append(part_tokens.input_ids)
                part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_atts.append(part_tokens.attention_mask)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]).to(
                    self.device))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_tokens.append(img_target[idx].unsqueeze(0))
                wrap_embeds.append(img_embeds[idx].unsqueeze(0))
                wrap_atts.append(atts_img[idx].unsqueeze(0))
                wrap_im_mask.append(torch.ones_like(atts_img[idx].unsqueeze(0))
                    )
                temp_len += im_len
            if temp_len > self.max_length:
                break
        wrap_tokens = torch.cat(wrap_tokens, dim=1)
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_atts = torch.cat(wrap_atts, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)
        wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
        wrap_atts = wrap_atts[:, :self.max_length].to(self.device)
        wrap_target = wrap_target[:, :self.max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device)
        wrap_embeds_list.append(wrap_embeds)
        wrap_atts_list.append(wrap_atts)
        wrap_target_list.append(wrap_target)
        wrap_im_mask_list.append(wrap_im_mask)
    wrap_embeds = torch.cat(wrap_embeds_list)
    wrap_atts = torch.cat(wrap_atts_list)
    wrap_target = torch.cat(wrap_target_list)
    wrap_im_mask = torch.cat(wrap_im_mask_list)
    return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask
