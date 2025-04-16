def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    embs_list, tokenid_list = [], []
    for image_list, question in zip(batch_images, batch_prompt):
        img_embes_list = []
        for image in image_list:
            image_emb, _ = self.model.encode_img(image)
            img_embes_list.append(image_emb)
        embs, input_ids = self.chat.get_context_emb(question, img_embes_list)
        tokenid_list.append(input_ids.squeeze(0))
        embs_list.append(embs.squeeze(0))
    embs_list = torch.nn.utils.rnn.pad_sequence([embs.flip(dims=[0]) for
        embs in embs_list], batch_first=True, padding_value=0.0).to(self.device
        ).flip(dims=[1])
    attn_mask = torch.all(embs_list != 0, dim=-1)
    input_ids = torch.nn.utils.rnn.pad_sequence([token.flip(dims=[0]) for
        token in tokenid_list], batch_first=True, padding_value=0.0).to(self
        .device).flip(dims=[1])
    outputs = self.model.llama_model(inputs_embeds=embs_list,
        attention_mask=attn_mask, return_dict=True)
    logits = outputs['logits'][:, :-1].float()
    labels = input_ids[:, 1:]
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.model.llama_tokenizer(option,
            add_special_tokens=False, return_tensors='pt').input_ids.squeeze(0)
            )
    results = []
    for idx in range(labels.shape[0]):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
        start_index = non_zero_indices.max() - option_len + 1
        end_index = start_index + option_len
        if not np.all(labels[idx][start_index:end_index].detach().cpu().
            numpy() == batch_option_ids[idx].numpy()):
            import ipdb
            ipdb.set_trace()
        prob = F.softmax(logits[idx][start_index:end_index], dim=-1)
        rows = torch.arange(0, option_len)
        score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean(
            ).item()
        results.append(score)
    return results
