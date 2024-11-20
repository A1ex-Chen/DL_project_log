def forward(self, samples):
    image = samples['image']
    image = image.to(self.device)
    img_embeds, atts_img = self.encode_img(image)
    if hasattr(samples, 'question_split'):
        print('VQA Batch')
        vqa_prompt = '###Human: <Img><ImageHere></Img> '
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
            vqa_prompt)
    elif self.prompt_list:
        prompt = random.choice(self.prompt_list)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
    self.llama_tokenizer.padding_side = 'right'
    text = [(t + self.end_sym) for t in samples['text_input']]
    to_regress_tokens = self.llama_tokenizer(text, return_tensors='pt',
        padding='longest', truncation=True, max_length=self.max_txt_len,
        add_special_tokens=False).to(image.device)
    targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.
        input_ids == self.llama_tokenizer.pad_token_id, -100)
    empty_targets = torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
        dtype=torch.long).to(image.device).fill_(-100)
    targets = torch.cat([empty_targets, targets], dim=1)
    batch_size = img_embeds.shape[0]
    bos = torch.ones([batch_size, 1], dtype=to_regress_tokens.input_ids.
        dtype, device=to_regress_tokens.input_ids.device
        ) * self.llama_tokenizer.bos_token_id
    bos_embeds = self.llama_model.model.embed_tokens(bos)
    atts_bos = atts_img[:, :1]
    to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens
        .input_ids)
    inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds],
        dim=1)
    attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.
        attention_mask], dim=1)
    with self.maybe_autocast():
        outputs = self.llama_model(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, return_dict=True, labels=targets)
    loss = outputs.loss
    return {'loss': loss}
