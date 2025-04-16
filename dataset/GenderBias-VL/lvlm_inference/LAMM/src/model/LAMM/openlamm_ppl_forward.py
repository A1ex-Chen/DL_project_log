def ppl_forward(self, inputs):
    assert self.vision_type == inputs['vision_type'
        ], '{} expected but {} given'.format(self.valid_type, inputs[
        'vision_type'])
    task_type = inputs['task_type']
    vision_paths = inputs['vision_paths']
    if self.vision_type == 'image':
        vision_embeds, _ = self.encode_image(vision_paths)
    elif self.vision_type == 'pcl':
        vision_embeds, _ = self.encode_pcl(vision_paths)
    else:
        raise ValueError('vision type [{}] not supported'.format(self.
            vision_type))
    output_texts = inputs['output_texts']
    input_ids, target_ids, attention_mask = process_batch_instance(self.
        llama_tokenizer, output_texts, self.max_tgt_len, self.vision_type,
        self.conv_template)
    inputs_embeds, targets, attention_mask = self.prompt_wrap(vision_embeds,
        input_ids, target_ids, attention_mask, self.use_system, task_type)
    outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=
        attention_mask, return_dict=True, labels=targets, use_cache=not
        self.use_flash_attn)
    logits = outputs.logits
    return logits, targets
