@torch.no_grad()
def generate(self, samples, use_nucleus_sampling=False, num_beams=5,
    max_length=256, min_length=1, top_p=0.9, repetition_penalty=1.5,
    length_penalty=1.0, num_captions=1, temperature=1):
    if 'prompt' in samples.keys():
        prompt = samples['prompt']
    else:
        prompt = self.prompt
    image = samples['image']
    bs = image.size(0)
    if isinstance(prompt, str):
        prompt = [prompt] * bs
    else:
        assert len(prompt
            ) == bs, 'The number of prompts must be equal to the batch size.'
    if 'ocr_tokens' in samples.keys() and '{}' in prompt[0]:
        prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i,
            p in enumerate(prompt)]
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
        inputs_t5, atts_t5 = [], []
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
            frame_inputs_t5 = self.t5_proj(frame_query_output.
                last_hidden_state[:, :query_tokens.size(1), :])
            frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=
                torch.long).to(image.device)
            inputs_t5.append(frame_inputs_t5)
            atts_t5.append(frame_atts_t5)
        inputs_t5 = torch.cat(inputs_t5, dim=1)
        atts_t5 = torch.cat(atts_t5, dim=1)
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
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :
            query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image
            .device)
    input_tokens = self.t5_tokenizer(prompt, padding='longest',
        return_tensors='pt').to(image.device)
    encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
    with self.maybe_autocast(dtype=torch.bfloat16):
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        outputs = self.t5_model.generate(inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts, do_sample=use_nucleus_sampling,
            top_p=top_p, temperature=temperature, num_beams=num_beams,
            max_new_tokens=max_length, min_length=min_length,
            repetition_penalty=repetition_penalty, length_penalty=
            length_penalty, num_return_sequences=num_captions)
        output_text = self.t5_tokenizer.batch_decode(outputs,
            skip_special_tokens=True)
    return output_text
