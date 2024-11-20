def preprocess_single(sample, audio_ext, text_ext, max_len, audio_cfg,
    tmodel, class_index_dict, data_filling, data_truncating,
    text_augment_selection):
    """
    Preprocess a single sample for wdsdataloader.
    """
    audio_data, orig_sr = sample[audio_ext]
    audio_data = int16_to_float32_torch(float32_to_int16_torch(audio_data[0]))
    sample = get_audio_features(sample, audio_data, max_len,
        data_truncating, data_filling, audio_cfg)
    del sample[audio_ext]
    json_dict_raw = sample[text_ext]
    texts = select_text(json_dict_raw, text_augment_selection)
    sample['full_text'] = texts
    if isinstance(texts, list) and isinstance(texts[0], str) and len(texts
        ) > 1:
        texts = random.choice(texts)
    sample['raw_text'] = texts
    sample['text'] = tokenizer(texts, tmodel=tmodel)
    if class_index_dict is not None:
        class_labels = np.zeros(len(class_index_dict))
        class_labels[np.in1d(list(class_index_dict.keys()), json_dict_raw[
            'tag'])] = 1
        sample['class_label'] = torch.tensor(class_labels).float()
    del sample[text_ext]
    sample['audio_name'] = sample['__key__'].split('/')[-1] + '.' + audio_ext
    sample['text_name'] = sample['__key__'].split('/')[-1] + '.' + text_ext
    sample['audio_orig_sr'] = orig_sr
    return sample
