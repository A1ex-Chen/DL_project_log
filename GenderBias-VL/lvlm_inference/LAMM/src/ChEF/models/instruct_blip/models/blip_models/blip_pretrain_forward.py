def forward(self, samples):
    """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images. Default: H=224, W=224.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
                - epoch (int): The current epoch.
                - iters (int): The current iteration.
                - num_iters_per_epoch (int): The number of iterations per epoch.

        Returns:
            BlipOutput: A BlipOutput object containing loss and intermediate output. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_pretrain", "base")
            >>> images = torch.randn(4, 3, 224, 224)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> samples = {"image": images, "text_input": text_input, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm', 'loss_lm'])

            >>> output.intermediate_output.keys()
            odict_keys(['image_embeds', 'text_embeds', 'image_embeds_m', 'text_embeds_m', 'encoder_output', 'encoder_output_neg', 'itm_logits', 'itm_labels', 'decoder_output', 'decoder_labels'])
            >>> output.intermediate_output.image_embeds.shape
            >>> # shape: (batch_size, num_patches, embed_dim)
            torch.Size([4, 197, 768])
            >>> output.intermediate_output.text_embeds.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.image_embeds_m.shape
            >>> # shape: (batch_size, num_patches, embed_dim)
            torch.Size([4, 197, 768])
            >>> output.intermediate_output.text_embeds_m.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.itm_logits.shape
            >>> # shape: (batch_size * 3, 2)
            torch.Size([12, 2])
            >>> output.intermediate_output.itm_labels.shape
            >>> # shape: (batch_size * 3,)
            torch.Size([12])
            >>> output.intermediate_output.encoder_output.last_hidden_state.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.encoder_output_m.last_hidden_state.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.decoder_output.logits.shape
            >>> # shape: (batch_size, max_txt_len, vocab_size)
            torch.Size([4, 30, 30524])
            >>> output.intermediate_output.decoder_labels.shape
            >>> # shape: (batch_size, max_txt_len)
            torch.Size([4, 30])
        """
    image = samples['image']
    caption = samples['text_input']
    alpha = self.alpha * self._rampup_factor(epoch=samples['epoch'], iters=
        samples['iters'], num_iters_per_epoch=samples['num_iters_per_epoch'])
    with torch.no_grad():
        self.temp.clamp_(0.001, 0.5)
    image_embeds = self.visual_encoder.forward_features(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
    text = self.tokenizer(caption, padding='max_length', truncation=True,
        max_length=self.max_txt_len, return_tensors='pt').to(image.device)
    text_output = self.text_encoder.forward_text(text)
    text_embeds = text_output.last_hidden_state
    text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
    with torch.no_grad():
        self._momentum_update()
        image_embeds_m = self.visual_encoder_m(image)
        image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0,
            :]), dim=-1)
        image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.
            clone().detach()], dim=1)
        text_output_m = self.text_encoder_m.forward_text(text)
        text_embeds_m = text_output_m.last_hidden_state
        text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]),
            dim=-1)
        text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone()
            .detach()], dim=1)
        sim_i2t_m = image_feat_m @ text_feat_all / self.temp
        sim_t2i_m = text_feat_m @ image_feat_all / self.temp
        sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
        sim_targets.fill_diagonal_(1)
        sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha
            ) * sim_targets
        sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha
            ) * sim_targets
    sim_i2t = image_feat @ text_feat_all / self.temp
    sim_t2i = text_feat @ image_feat_all / self.temp
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,
        dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets,
        dim=1).mean()
    loss_itc = (loss_i2t + loss_t2i) / 2
    self._dequeue_and_enqueue(image_feat_m, text_feat_m)
    encoder_input_ids = text.input_ids.clone()
    encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
    bs = image.size(0)
    output_pos = self.text_encoder(encoder_input_ids, attention_mask=text.
        attention_mask, encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts, return_dict=True)
    with torch.no_grad():
        weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 0.0001
        weights_t2i.fill_diagonal_(0)
        weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 0.0001
        weights_i2t.fill_diagonal_(0)
    image_embeds_neg = []
    for b in range(bs):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        image_embeds_neg.append(image_embeds[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    text_ids_neg = []
    text_atts_neg = []
    for b in range(bs):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(encoder_input_ids[neg_idx])
        text_atts_neg.append(text.attention_mask[neg_idx])
    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)
    text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
    text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)
    image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
    image_atts_all = torch.cat([image_atts, image_atts], dim=0)
    output_neg = self.text_encoder(text_ids_all, attention_mask=
        text_atts_all, encoder_hidden_states=image_embeds_all,
        encoder_attention_mask=image_atts_all, return_dict=True)
    vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :],
        output_neg.last_hidden_state[:, 0, :]], dim=0)
    itm_logits = self.itm_head(vl_embeddings)
    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 *
        bs, dtype=torch.long)], dim=0).to(image.device)
    loss_itm = F.cross_entropy(itm_logits, itm_labels)
    decoder_input_ids = text.input_ids.clone()
    decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
    decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids ==
        self.tokenizer.pad_token_id, -100)
    decoder_output = self.text_decoder(decoder_input_ids, attention_mask=
        text.attention_mask, encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts, labels=decoder_targets,
        return_dict=True)
    loss_lm = decoder_output.loss
    return BlipOutput(loss=loss_itc + loss_itm + loss_lm, loss_itc=loss_itc,
        loss_itm=loss_itm, loss_lm=loss_lm, sims=BlipSimilarity(sim_i2t=
        sim_i2t, sim_t2i=sim_t2i, sim_i2t_m=sim_i2t_m, sim_t2i_m=sim_t2i_m,
        sim_i2t_targets=sim_i2t_targets, sim_t2i_targets=sim_t2i_targets),
        intermediate_output=BlipIntermediateOutput(image_embeds=
        image_embeds, text_embeds=text_embeds, image_embeds_m=
        image_embeds_m, text_embeds_m=text_embeds_m, encoder_output=
        output_pos, encoder_output_neg=output_neg, itm_logits=itm_logits,
        itm_labels=itm_labels, decoder_output=decoder_output,
        decoder_labels=decoder_targets))
