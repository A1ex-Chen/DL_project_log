def preparing_embedding(self, samples):
    if 'image' in samples:
        img_embeds, img_atts = self.encode_img(samples['image'])
    else:
        img_embeds = img_atts = None
    if 'conv_q' in samples:
        conv_q, conv_a = samples['conv_q'], samples['conv_a']
        connect_sym = samples['connect_sym'][0]
        conv_q = [q.split(connect_sym) for q in conv_q]
        conv_a = [a.split(connect_sym) for a in conv_a]
        conv_q = [[self.prompt_template.format(item) for item in items] for
            items in conv_q]
        cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[
            0] for q in conv_q])
        regress_token_ids, regress_atts, part_targets = (self.
            tokenize_conversation(conv_q, conv_a))
    else:
        if 'instruction_input' in samples:
            instruction = samples['instruction_input']
        elif self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = None
        if hasattr(self, 'chat_template') and self.chat_template:
            instruction = [self.prompt_template.format(instruct) for
                instruct in instruction]
        if 'length' in samples:
            bsz, pn, hs = img_embeds.shape
            img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts,
                instruction, samples['length'])
        else:
            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts,
                instruction)
        self.llama_tokenizer.padding_side = 'right'
        text = [(t + self.end_sym) for t in samples['answer']]
        regress_tokens = self.llama_tokenizer(text, return_tensors='pt',
            padding='longest', truncation=True, max_length=self.max_txt_len,
            add_special_tokens=False).to(self.device)
        regress_token_ids = regress_tokens.input_ids
        regress_atts = regress_tokens.attention_mask
        part_targets = regress_token_ids.masked_fill(regress_token_ids ==
            self.llama_tokenizer.pad_token_id, -100)
    regress_embeds = self.embed_tokens(regress_token_ids)
    return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets
