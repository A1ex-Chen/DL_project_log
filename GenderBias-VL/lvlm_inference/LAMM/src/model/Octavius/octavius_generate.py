def generate(self, inputs):
    """
        inputs = {
            'image_paths': optional,
            'mode': generation mode,
            'prompt': human input prompt,
            'max_tgt_len': generation length,
            'top_p': top_p,
            'temperature': temperature
            'modality_embeds': None or torch.tensor
            'modality_cache': save the image cache
        }
        """
    input_embeds, input_masks, prompt_embeds = (self.
        prepare_generation_embedding(inputs))
    stopping_criteria = StoppingCriteriaList([LAMMStoppingCriteria([[2277, 
        29937], [835], [1, 2]], input_embeds)])
    self.moe_set_gate(prompt_embeds, input_embeds.device)
    outputs = self.llama_model.generate(inputs_embeds=input_embeds,
        attention_mask=input_masks, max_new_tokens=inputs['max_tgt_len'],
        top_p=inputs['top_p'], temperature=inputs['temperature'], do_sample
        =True, use_cache=True, stopping_criteria=stopping_criteria)
    output_text = self.llama_tokenizer.batch_decode(outputs,
        skip_special_tokens=True)
    return output_text
