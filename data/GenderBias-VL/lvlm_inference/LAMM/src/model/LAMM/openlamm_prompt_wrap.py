def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask,
    use_system, task_type):
    """
        input_ids, target_ids, attention_mask: bsz x s2
        """
    input_ids = input_ids.to(self.device)
    target_ids = target_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    batch_size = img_embeds.shape[0]
    p_before = make_prompt_start(use_system=use_system, vision_type=self.
        vision_type, task_type=task_type, template=self.conv_template)
    if isinstance(p_before, list):
        p_before_tokens = [self.llama_tokenizer(p, return_tensors='pt',
            add_special_tokens=False).input_ids[0].to(self.device) for p in
            p_before]
        p_before_token_ids = rnn.pad_sequence(p_before_tokens, batch_first=
            True, padding_value=self.llama_tokenizer.pad_token_id)
        p_before_attn_mask = p_before_token_ids.ne(self.llama_tokenizer.
            pad_token_id)
    else:
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors=
            'pt', add_special_tokens=False).to(self.device)
        p_before_token_ids = p_before_tokens.input_ids.expand(batch_size, -1)
        p_before_attn_mask = p_before_tokens.attention_mask.expand(batch_size,
            -1)
    p_before_embeds = self.embed_tokens(p_before_token_ids)
    p_after_embeds = self.embed_tokens(input_ids).expand(batch_size, -1, -1)
    bos = torch.ones([batch_size, 1], dtype=p_before_token_ids.dtype,
        device=p_before_token_ids.device) * self.llama_tokenizer.bos_token_id
    bos_embeds = self.embed_tokens(bos)
    inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds,
        p_after_embeds], dim=1)
    empty_targets = torch.ones([batch_size, 1 + p_before_embeds.size()[1] +
        self.num_vision_token], dtype=torch.long).to(self.device).fill_(-100)
    targets = torch.cat([empty_targets, target_ids], dim=1)
    assert inputs_embeds.size()[1] == targets.size()[1]
    atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(self.device)
    atts_img = torch.ones([batch_size, self.num_vision_token], dtype=torch.long
        ).to(self.device)
    attention_mask = torch.cat([atts_bos, p_before_attn_mask, atts_img,
        attention_mask], dim=1)
    assert attention_mask.size() == targets.size()
    return inputs_embeds, targets, attention_mask
