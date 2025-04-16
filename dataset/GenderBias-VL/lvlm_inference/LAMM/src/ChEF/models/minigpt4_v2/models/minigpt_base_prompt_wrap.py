def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
    if prompts is None or len(prompts) == 0:
        return img_embeds, atts_img
    elif img_embeds is None:
        self.llama_tokenizer.padding_side = 'right'
        prompt_tokens = self.llama_tokenizer(prompts, return_tensors='pt',
            padding='longest', add_special_tokens=False).to(self.device)
        prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
        atts_prompt = prompt_tokens.attention_mask
        return prompt_embeds, atts_prompt
    else:
        emb_lists = []
        if isinstance(prompts, str):
            prompts = [prompts] * len(img_embeds)
        for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds,
            prompts)):
            pn = each_img_embed.shape[-2]
            if lengths is not None:
                each_img_embed = each_img_embed.reshape(-1, each_img_embed.
                    shape[-1])
                each_img_embed = each_img_embed[:lengths[idx] * pn]
            p_segs = each_prompt.split('<ImageHere>')
            interleave_emb = []
            for idx, seg in enumerate(p_segs[:-1]):
                p_tokens = self.llama_tokenizer(seg, return_tensors='pt',
                    add_special_tokens=False).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                interleave_emb.append(torch.cat([p_embed, each_img_embed[
                    None][:, idx * pn:(idx + 1) * pn]], dim=1))
            wrapped_emb = torch.cat(interleave_emb, dim=1)
            p_tokens = self.llama_tokenizer(p_segs[-1], return_tensors='pt',
                add_special_tokens=False).to(img_embeds.device)
            p_embed = self.embed_tokens(p_tokens.input_ids)
            wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
            emb_lists.append(wrapped_emb)
        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.
            pad_token_id, device=img_embeds.device))
        max_length = max(emb_lens) if max(emb_lens
            ) < self.max_context_len else self.max_context_len
        wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch
            .int, device=img_embeds.device)
        for i, emb in enumerate(emb_lists):
            length = emb_lens[i] if emb_lens[i
                ] < self.max_context_len else self.max_context_len
            wrapped_embs[i, :length] = emb[:, :length]
            wrapped_atts[i, :length] = 1
        return wrapped_embs, wrapped_atts
