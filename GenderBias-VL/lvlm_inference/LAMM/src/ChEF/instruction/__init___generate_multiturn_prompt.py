def generate_multiturn_prompt(self, batch, prompt_idx_list, prefix_list, **
    kwargs):
    prompt_idx_list = [(min(prompt_idx, 1) if prompt_idx is not None else
        None) for prompt_idx in prompt_idx_list]
    multi_turn_batch_index = []
    multi_turn_batch_tmp_index = 0
    image_path, questions = [], []
    return_dict = {key: [] for key in kwargs.keys() if kwargs[key] is not None}
    for i, (prompt_idx, prefix) in enumerate(zip(prompt_idx_list, prefix_list)
        ):
        if prompt_idx is None:
            multi_turn_batch_index.append(None)
            continue
        image_path.append(batch['image_path'][i])
        questions.append(self.prompt[prompt_idx].format(prefix=prefix))
        multi_turn_batch_index.append(multi_turn_batch_tmp_index)
        multi_turn_batch_tmp_index += 1
        for key in return_dict.keys():
            return_dict[key].append(kwargs[key][i])
    return_dict['batch_images'] = image_path
    return_dict['batch_prompt'] = questions
    return_dict['multi_turn_batch_index'] = multi_turn_batch_index
    return return_dict
