def prepare_generation_embedding(self, inputs):
    """prepare for generation

        :param class inputs: model
        :return Dict: generation input
        """
    prompt_list = inputs['prompt']
    if len(inputs['modality_embeds']) == 1:
        feature_embeds = inputs['modality_embeds'][0]
    else:
        feature_embeds = self.extract_multimodal_feature(inputs)
        inputs['modality_embeds'].append(feature_embeds)
    eov = VISION_TAGS['eov'][self.vision_type]
    batch_size = feature_embeds.shape[0]
    p_before = make_prompt_start(use_system=False, vision_type=[self.
        vision_type for _ in range(batch_size)], task_type=['normal' for _ in
        range(batch_size)], template=self.conv_template)
    p_before_tokens = self.llama_tokenizer(p_before, return_tensors='pt',
        add_special_tokens=False).to(self.device)
    p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens
        .input_ids).expand(batch_size, -1, -1)
    p_after_texts = [(f'{eov} ' + prompt +
        f"""
{self.conv_template.sep} {self.conv_template.roles[1]}:""") for
        prompt in prompt_list]
    p_after_tokens = self.llama_tokenizer(p_after_texts, padding='longest',
        return_length=True, add_special_tokens=False, return_tensors='pt').to(
        self.device)
    p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
    p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens
        .input_ids)
    bos = torch.ones([batch_size, 1], dtype=p_before_tokens.input_ids.dtype,
        device=p_before_tokens.input_ids.device
        ) * self.llama_tokenizer.bos_token_id
    bos_embeds = self.llama_model.model.model.embed_tokens(bos)
    inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds,
        p_after_embeds], dim=1)
    tokens_len = inputs_embeds.shape[1] - p_after_masks_len
    new_inputs_embeds = torch.zeros_like(inputs_embeds)
    inputs_embeds_masks = torch.zeros(inputs_embeds.shape[:-1], dtype=torch
        .int64, device=self.device)
    for idx in range(batch_size):
        inputs_embeds_masks[idx, -tokens_len[idx]:] = 1
        new_inputs_embeds[idx, -tokens_len[idx]:, :] = inputs_embeds[idx, :
            tokens_len[idx], :]
        new_inputs_embeds[idx, :-tokens_len[idx], :] = inputs_embeds[idx,
            tokens_len[idx]:, :]
    return inputs_embeds, inputs_embeds_masks, p_after_embeds
