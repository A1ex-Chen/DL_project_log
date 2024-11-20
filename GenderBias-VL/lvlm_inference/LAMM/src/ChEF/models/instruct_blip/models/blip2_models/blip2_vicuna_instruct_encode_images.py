@torch.no_grad()
def encode_images(self, images):
    bs = images.size(0)
    query_tokens = self.query_tokens.expand(bs, -1, -1)
    if images.dim() == 5:
        inputs_llm, atts_llm = [], []
        for j in range(images.size(2)):
            this_frame = images[:, :, j, :, :]
            with self.maybe_autocast():
                frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
            frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long
                ).to(images.device)
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
            image_embeds = self.ln_vision(self.visual_encoder(images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            images.device)
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :
            query_tokens.size(1), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            images.device)
