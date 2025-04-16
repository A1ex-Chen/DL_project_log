def generate_singleturn_ppl_prompt(self, prompts, batch, batch_options, **
    kwargs):
    answer_template = self.answer_template
    batch_size = len(batch_options)
    batch_ppl_len = [len(batch_option) for batch_option in batch_options]
    ppl_len = sum(batch_ppl_len)
    ppl_batch_mask = np.zeros((batch_size, ppl_len))
    ppl_batch_mask_tmp_index = 0
    image_path, answers, questions, options = [], [], [], []
    return_dict = {key: [] for key in kwargs.keys() if kwargs[key] is not None}
    for i in range(batch_size):
        answers += [answer_template.format(option=option) for option in
            batch_options[i]]
        options += batch_options[i]
        new_len = len(batch_options[i])
        questions += [prompts[i]] * new_len
        image_path += [batch['image_path'][i]] * new_len
        ppl_batch_mask[i][ppl_batch_mask_tmp_index:ppl_batch_mask_tmp_index +
            new_len] = 1
        ppl_batch_mask_tmp_index += new_len
        for key in return_dict.keys():
            return_dict[key] += [kwargs[key][i]] * new_len
    ppl_batch_mask = np.array(ppl_batch_mask, dtype=bool)
    return_dict['batch_images'] = image_path
    return_dict['batch_prompt'] = questions
    return_dict['batch_answers'] = answers
    return_dict['batch_options'] = options
    return_dict['ppl_batch_mask'] = ppl_batch_mask
    return return_dict
