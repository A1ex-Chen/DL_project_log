def select_text(json_dict_raw, text_augment_selection):
    if text_augment_selection is None or text_augment_selection == 'none':
        texts = json_dict_raw['text']
    elif text_augment_selection == 'all':
        if 'text_augment_all' in json_dict_raw.keys():
            texts = json_dict_raw['text_augment_all']
        else:
            texts = json_dict_raw['text']
    elif text_augment_selection == 'augment_only':
        if 'text_augment_all' in json_dict_raw.keys():
            if json_dict_raw['text_augment_t5'] is None:
                texts = json_dict_raw['text']
            else:
                texts = json_dict_raw['text_augment_t5']
        else:
            texts = json_dict_raw['text']
    else:
        raise NotImplementedError(
            f'text_augment_selection {text_augment_selection} not implemented')
    return texts
