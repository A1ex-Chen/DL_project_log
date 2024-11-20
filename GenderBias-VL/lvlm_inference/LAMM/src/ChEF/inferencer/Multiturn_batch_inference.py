def batch_inference(self, model, batch, **kwargs):
    multiturn_prefix = batch['multi_turn_prefix']
    turn_num = max([len(item) for item in multiturn_prefix])
    predictions = [copy_batch_dict(batch, i) for i in range(len(
        multiturn_prefix))]
    for item in predictions:
        item['turn_answer'] = []
    for turn_idx in range(turn_num):
        prompt_idx_list = [(item_prefix[turn_idx]['prompt_idx'] if len(
            item_prefix) > turn_idx else None) for item_prefix in
            multiturn_prefix]
        prefix_list = [(item_prefix[turn_idx]['prefix'] if len(item_prefix) >
            turn_idx else None) for item_prefix in multiturn_prefix]
        return_dict = self.instruction_handler.generate_multiturn_prompt(batch,
            prompt_idx_list, prefix_list)
        outputs = model.batch_generate(max_new_tokens=self.max_new_tokens,
            **return_dict)
        multi_turn_batch_index = return_dict['multi_turn_batch_index']
        for i in range(len(multiturn_prefix)):
            answer_index = multi_turn_batch_index[i]
            if answer_index is None:
                continue
            predictions[i]['turn_answer'].append(dict(prompt_idx=
                prompt_idx_list[i], question=return_dict['batch_prompt'][
                answer_index], answer=outputs[answer_index]))
    return predictions
