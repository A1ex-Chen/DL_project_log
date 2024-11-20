def generate_multiturn_ppl_prompt(self, batch, prompt_idx_list, prefix_list,
    batch_options, **kwargs):
    prompt_idx_list = [(min(prompt_idx, 1) if prompt_idx is not None else
        None) for prompt_idx in prompt_idx_list]
    prompt_list = []
    for prompt_idx, prefix in zip(prompt_idx_list, prefix_list):
        if prompt_idx is None:
            prompt_list.append(None)
            continue
        prompt_list.append(self.prompt[prompt_idx].format(prefix=prefix))
    answer_template_list = [(self.answer_template[prompt_idx] if prompt_idx
         is not None else None) for prompt_idx in prompt_idx_list]
    multi_turn_batch_index = []
    multi_turn_batch_tmp_index = 0
    image_path, answers, questions, options = [], [], [], []
    return_dict = {key: [] for key in kwargs.keys()}
    for i, (prompt, answer_template, sample_option) in enumerate(zip(
        prompt_list, answer_template_list, batch_options)):
        if prompt is None:
            multi_turn_batch_index.append(None)
            continue
        answers += [answer_template.format(option=option) for option in
            sample_option]
        options += sample_option
        new_len = len(sample_option)
        questions += [prompt] * new_len
        image_path += [batch['image_path'][i]] * new_len
        multi_turn_batch_index.append([i for i in range(
            multi_turn_batch_tmp_index, multi_turn_batch_tmp_index + new_len)])
        multi_turn_batch_tmp_index += new_len
        for key in return_dict.keys():
            return_dict[key] += [kwargs[key][i]] * new_len
    return_dict['batch_images'] = image_path
    return_dict['batch_prompt'] = questions
    return_dict['batch_answers'] = answers
    return_dict['batch_options'] = options
    return_dict['ppl_batch_index'] = multi_turn_batch_index
    return return_dict
