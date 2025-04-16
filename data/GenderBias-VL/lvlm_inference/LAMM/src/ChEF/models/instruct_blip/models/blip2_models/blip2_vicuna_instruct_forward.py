def forward(self, samples):
    image = samples['image']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    bs = image.size(0)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(samples['text_input'], padding=
            'longest', truncation=True, max_length=self.max_txt_len,
            return_tensors='pt').to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
            dim=1)
        query_output = self.Qformer.bert(text_Qformer.input_ids,
            attention_mask=Qformer_atts, query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
    else:
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
    inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :
        query_tokens.size(1), :])
    atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image
        .device)
    self.llm_tokenizer.padding_side = 'right'
    self.llm_tokenizer.truncation_side = 'left'
    text_input_tokens = self.llm_tokenizer(samples['text_input'],
        return_tensors='pt', padding='longest', truncation=True, max_length
        =self.max_txt_len).to(image.device)
    self.llm_tokenizer.truncation_side = 'right'
    text_output_tokens = self.llm_tokenizer([(t + self.llm_tokenizer.
        eos_token) for t in samples['text_output']], return_tensors='pt',
        padding='longest', truncation=True, max_length=self.max_output_txt_len
        ).to(image.device)
    llm_tokens, input_part_targets_len = self.concat_text_input_output(
        text_input_tokens.input_ids, text_input_tokens.attention_mask,
        text_output_tokens.input_ids, text_output_tokens.attention_mask)
    targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] ==
        self.llm_tokenizer.pad_token_id, -100)
    for i, l in enumerate(input_part_targets_len):
        targets[i][:l] = -100
    empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.
        device).fill_(-100)
    targets = torch.cat([empty_targets, targets], dim=1)
    inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens[
        'input_ids'])
    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
    with self.maybe_autocast():
        outputs = self.llm_model(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, return_dict=True, labels=targets)
    return outputs, targets
