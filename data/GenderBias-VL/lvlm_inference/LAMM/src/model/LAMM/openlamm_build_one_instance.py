def build_one_instance(tokenizer, conversation, vision_type='image',
    template=conversations.default_conversation):
    """build one instance for training; text part

    :param class tokenizer: text tokenizer
    :param list conversation: list of conversation
    :param str vision_type: type of vision data, defaults to 'image'
    :raises Exception: Exception if wrong role included
    :return list: conversation text list, input token ids, target token ids
    """
    pos = VISION_TAGS['pos'][vision_type]
    eov = VISION_TAGS['eov'][vision_type]
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0:
            assert role == 'human'
            turn['value'] = turn['value'].replace(f'{pos}\n', '').replace(
                f'\n{pos}', '')
            text = f'{eov} ' + turn['value'] + '\n{} {}:'.format(template.
                sep, template.roles[1])
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)
        elif role == 'human':
            text = '{}: {}\n{} {}:'.format(template.roles[0], turn['value'],
                template.sep, template.roles[1])
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)
        elif role == 'gpt':
            text = turn['value'] + '\n{}'.format(template.sep2 if template.
                sep2 is not None else template.sep)
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += one_input_id
        else:
            raise Exception(f'{role} is a Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids
