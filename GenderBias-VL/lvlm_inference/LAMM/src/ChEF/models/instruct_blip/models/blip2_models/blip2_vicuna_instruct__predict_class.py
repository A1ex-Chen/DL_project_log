def _predict_class(self, samples, candidates, n_segments=1):
    image = samples['image']
    prompt = samples['prompt']
    bs = image.size(0)
    if isinstance(prompt, str):
        prompt = [prompt] * bs
    else:
        assert len(prompt
            ) == bs, 'The number of prompts must be equal to the batch size.'
    if 'text_input' in samples.keys():
        if type(samples['text_input'][0]) == list:
            prompt = [prompt[i].format(*samples['text_input'][i]) for i in
                range(len(prompt))]
        else:
            prompt = [prompt[i].format(samples['text_input'][i]) for i in
                range(len(prompt))]
    if 'context' in samples.keys() and samples['context'] != '':
        prompt = [f"context: {samples['context'][i]}. {prompt[i]}" for i in
            range(len(prompt))]
    if 'history' in samples.keys() and samples['history'][0] != '':
        prompt = [f"dialog history: {samples['history'][i]}\n{prompt[i]}" for
            i in range(len(prompt))]
    if 'caption' in samples.keys() and samples['caption'][0] != '':
        prompt = [
            f'This image has the caption "{samples[\'caption\'][i]}". {prompt[i]}'
             for i in range(len(prompt))]
    query_tokens = self.query_tokens.expand(bs, -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(prompt, padding='longest', truncation
            =True, max_length=self.max_txt_len, return_tensors='pt').to(image
            .device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
            dim=1)
    if image.dim() == 5:
        inputs_llm, atts_llm = [], []
        for j in range(image.size(2)):
            this_frame = image[:, :, j, :, :]
            with self.maybe_autocast():
                frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=
                    torch.long).to(image.device)
            if self.qformer_text_input:
                frame_query_output = self.Qformer.bert(text_Qformer.
                    input_ids, attention_mask=Qformer_atts, query_embeds=
                    query_tokens, encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts, return_dict=True)
            else:
                frame_query_output = self.Qformer.bert(query_embeds=
                    query_tokens, encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts, return_dict=True)
            frame_inputs_llm = self.llm_proj(frame_query_output.
                last_hidden_state[:, :query_tokens.size(1), :])
            frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype
                =torch.long).to(image.device)
            inputs_llm.append(frame_inputs_llm)
            atts_llm.append(frame_atts_llm)
        inputs_llm = torch.cat(inputs_llm, dim=1)
        atts_llm = torch.cat(atts_llm, dim=1)
    else:
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device)
        if self.qformer_text_input:
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
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            image.device)
    self.llm_tokenizer.padding_side = 'right'
    self.llm_tokenizer.truncation_side = 'left'
    text_input_tokens = self.llm_tokenizer(prompt, return_tensors='pt',
        padding='longest').to(image.device)
    empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.
        device).fill_(-100)
    self.llm_tokenizer.truncation_side = 'right'
    n_cands = len(candidates)
    with self.maybe_autocast(dtype=torch.bfloat16):
        all_losses = []
        for n in range(n_segments):
            seg_len = n_cands // n_segments
            if n == n_segments - 1:
                seg_len = n_cands - seg_len * (n_segments - 1)
            start_i = n * (n_cands // n_segments)
            end_i = start_i + seg_len
            this_output_tokens = self.llm_tokenizer(candidates[start_i:
                end_i], return_tensors='pt', padding='longest').to(image.device
                )
            this_input_tokens_ids = (text_input_tokens.input_ids.
                repeat_interleave(seg_len, dim=0))
            this_input_tokens_atts = (text_input_tokens.attention_mask.
                repeat_interleave(seg_len, dim=0))
            this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
            this_output_tokens_atts = this_output_tokens.attention_mask.repeat(
                bs, 1)
            this_llm_tokens, this_input_targets_len = (self.
                concat_text_input_output(this_input_tokens_ids,
                this_input_tokens_atts, this_output_tokens_ids,
                this_output_tokens_atts))
            this_llm_input_ids = this_llm_tokens['input_ids']
            this_llm_atts = this_llm_tokens['attention_mask']
            inputs_embeds = self.llm_model.get_input_embeddings()(
                this_llm_input_ids)
            inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len,
                dim=0), inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len,
                dim=0), this_llm_atts], dim=1)
            this_targets = this_llm_input_ids.masked_fill(
                this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
            for i, l in enumerate(this_input_targets_len):
                this_targets[i][:l] = -100
            this_targets = torch.cat([empty_targets.repeat_interleave(
                seg_len, dim=0), this_targets], dim=1)
            outputs = self.llm_model(inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, return_dict=True, labels=
                this_targets, reduction='none')
            loss = outputs.loss
            loss = loss.reshape(bs, seg_len)
            all_losses.append(loss)
        all_losses = torch.cat(all_losses, dim=-1)
        output_class_ranks = torch.argsort(all_losses, dim=-1)
    return output_class_ranks
