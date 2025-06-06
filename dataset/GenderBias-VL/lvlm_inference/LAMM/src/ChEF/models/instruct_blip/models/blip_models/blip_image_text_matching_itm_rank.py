def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head=
    'itm'):
    encoder_input_ids = encoder_input_ids.clone()
    encoder_input_ids = encoder_input_ids[:, 3:]
    text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id
        ).long()
    if match_head == 'itm':
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        output = self.text_encoder(encoder_input_ids, attention_mask=
            text_attention_mask, encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts, return_dict=True)
        itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
        itm_output = F.softmax(itm_output, dim=1)[:, 1]
        return itm_output
    elif match_head == 'itc':
        encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
        text_output = self.text_encoder(encoder_input_ids, attention_mask=
            text_attention_mask, return_dict=True, mode='text')
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]),
            dim=-1)
        text_feat = F.normalize(self.text_proj(text_output.
            last_hidden_state[:, 0, :]), dim=-1)
        sim = image_feat @ text_feat.t()
        return sim
