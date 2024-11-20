def preprocess(sample, audio_ext, text_ext, max_len, audio_cfg,
    class_index_dict=None, data_filling='pad', data_truncating='rand_trunc',
    text_augment_selection=None):
    """
    Preprocess a single sample for wdsdataloader.
    """
    audio_data, orig_sr = sf.read(io.BytesIO(sample[audio_ext]))
    audio_data = int16_to_float32(float32_to_int16(audio_data))
    audio_data = torch.tensor(audio_data).float()
    sample = get_audio_features(sample, audio_data, max_len,
        data_truncating, data_filling, audio_cfg)
    del sample[audio_ext]
    try:
        json_dict_raw = json.loads(sample[text_ext].decode('utf-8'))
    except:
        print('sample[__url__]:', sample['__url__'])
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
    sample['full_text'] = texts
    if isinstance(texts, list) and isinstance(texts[0], str) and len(texts
        ) > 1:
        texts = random.choice(texts)
    sample['raw_text'] = texts
    sample['text'] = tokenizer(texts)
    if class_index_dict is not None:
        sample['class_label'] = np.zeros(len(class_index_dict.keys()))
        for x in json_dict_raw['tag']:
            sample['class_label'][class_index_dict[x]] = 1
        sample['class_label'] = torch.tensor(sample['class_label']).float()
    del sample[text_ext]
    sample['audio_name'] = sample['__key__'].split('/')[-1] + '.' + audio_ext
    sample['text_name'] = sample['__key__'].split('/')[-1] + '.' + text_ext
    sample['audio_orig_sr'] = orig_sr
    return sample
