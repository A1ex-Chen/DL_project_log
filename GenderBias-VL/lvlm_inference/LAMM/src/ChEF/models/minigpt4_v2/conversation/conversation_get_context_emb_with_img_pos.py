def get_context_emb_with_img_pos(self, prompt, img_list):
    img_embeds_start, img_embeds_end = 0, 0
    img_embeds_len = 0
    prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list
        ) + 1, 'Unmatched numbers of image placeholders and images.'
    seg_tokens = [self.model.llama_tokenizer(seg, return_tensors='pt',
        add_special_tokens=i == 0).to(self.device).input_ids for i, seg in
        enumerate(prompt_segs)]
    seg_embs = [self.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    img_embeds_start = seg_embs[0].shape[1]
    img_embeds_len = img_list[0].shape[1]
    img_embeds_end = img_embeds_start + img_embeds_len
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair
        ] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    img_ids = [torch.zeros(1, img.shape[1]).to(self.device) for img in img_list
        ]
    token_ids = [token for pair in zip(seg_tokens[:-1], img_ids) for token in
        pair] + [seg_tokens[-1]]
    token_ids = torch.cat(token_ids, dim=1)
    return (mixed_embs, token_ids, img_embeds_start, img_embeds_end,
        img_embeds_len)
