def prepare_few_shot_embeds(self, samples):
    this_n_fs = random.choices(list(range(self.num_few_shot_examples + 1)),
        weights=[1 - self.few_shot_prob] + [self.few_shot_prob / self.
        num_few_shot_examples] * self.num_few_shot_examples)[0]
    if this_n_fs == 0:
        return None, None
    images = []
    text_input = []
    for sample in samples:
        for n in range(this_n_fs):
            images.append(sample['image'][n])
            text_input.append(sample['text_input'][n])
    images = torch.stack(images, dim=0)
    image = images
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(text_input, padding='longest',
            truncation=True, max_length=self.max_txt_len, return_tensors='pt'
            ).to(image.device)
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
    inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :
        query_tokens.size(1), :])
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.
        device)
    with self.maybe_autocast(dtype=torch.bfloat16):
        input_tokens = self.t5_tokenizer(text_input, padding='longest',
            truncation=True, max_length=self.max_txt_len, return_tensors='pt'
            ).to(image.device)
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
    if this_n_fs > 1:
        encoder_atts = encoder_atts.reshape(encoder_atts.size(0) //
            this_n_fs, encoder_atts.size(1) * this_n_fs)
        inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) //
            this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2)
            )
    return inputs_embeds, encoder_atts
