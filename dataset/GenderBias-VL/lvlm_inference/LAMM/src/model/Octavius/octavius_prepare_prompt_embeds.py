def prepare_prompt_embeds(self, vision_embeds, vision_mask, output_texts,
    vision_type, task_type):
    output_texts = self.reconstruct_gt_input(output_texts, task_type)
    input_ids, target_ids, attention_mask = process_batch_instance(self.
        llama_tokenizer, output_texts, self.max_tgt_len, vision_type, self.
        conv_template)
    inputs_embeds, targets, attention_mask = self.prompt_wrap(vision_embeds,
        input_ids, target_ids, attention_mask, vision_mask, self.use_system,
        vision_type, task_type)
    return inputs_embeds, targets, attention_mask
