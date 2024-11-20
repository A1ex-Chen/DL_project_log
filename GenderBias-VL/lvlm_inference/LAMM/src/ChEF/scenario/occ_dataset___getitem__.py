def __getitem__(self, idx):
    item = self.data[idx]
    question = item['query']
    question = clean_question(question, self.generative)
    img_path = os.path.join(self.base_data_path, item['image'])
    gt_choice = item['gt_choice']
    gt_answers = item['gt_choices'][gt_choice]
    choices = item['gt_choices']
    id = str(item['id']) if 'id' in item else str(idx)
    res_dict = {'id': id, 'image_path': img_path, 'question': question,
        'gt_answers': gt_answers, 'gt_choice': gt_choice, 'choices': choices}
    if self.generative:
        res_dict['options'] = choices
        res_dict['gt_answers'] = choices[res_dict['gt_choice']]
    else:
        res_dict['options'] = get_options(choices, self.option_content)
        res_dict['gt_answers'] = '(' + OPTION[res_dict['gt_choice']] + ')'
    if self.map_type != None:
        map_text = ''
        map_template = 'If the answer is "{}", you need to output "{}". '
        if self.map_type == 'unnatural':
            if self.map_id == 0:
                option_map = res_dict['options'][1:] + res_dict['options'][:1]
            else:
                option_map = res_dict['options'][-1:] + res_dict['options'][:-1
                    ]
        else:
            option_map = self.option_map
        for opid, opt in enumerate(res_dict['options']):
            map_text += map_template.format(opt + ')', option_map[opid])
        res_dict['question'] += map_text
        res_dict['options'] = option_map[:len(res_dict['options'])]
    return res_dict
