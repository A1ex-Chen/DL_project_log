def data_wrapper(dataset, tokenizer, decoder_start_token_id):
    batch_size = len(dataset)
    new_dataset = {'gt': [d['gt'] for d in dataset], 'gt_concepts': [d[
        'gt_concepts'] for d in dataset]}
    _PAD = tokenizer.pad_token_id
    _EOS = tokenizer.eos_token_id
    _BOS = decoder_start_token_id
    max_concept_len = max([len(d['concept_set']) for d in dataset])
    concept_set_input = np.full((batch_size, max_concept_len), _PAD, dtype=
        np.int64)
    cls_on_input = np.full((batch_size, max_concept_len), 0, dtype=np.int64)
    for i, d in enumerate(dataset):
        concept_set_input[i, :len(d['concept_set'])] = d['concept_set']
        cls_on_input[i, :d['cls_on_input'].shape[0]] = d['cls_on_input']
    new_dataset['input_ids'] = torch.from_numpy(concept_set_input)
    new_dataset['cls_on_input'] = torch.from_numpy(cls_on_input)
    new_dataset['attention_mask'] = (new_dataset['input_ids'] != _PAD).float()
    copy_pos = np.zeros((batch_size, 5, max_concept_len), dtype=np.float32)
    for i, d in enumerate(dataset):
        copy_pos[i, :, :d['copy_pos'].shape[-1]] = d['copy_pos']
    new_dataset['copy_pos'] = torch.from_numpy(copy_pos)
    concept_cls = np.zeros((batch_size, 5), dtype=np.int64)
    for i, d in enumerate(dataset):
        concept_cls[i, :len(d['concept_cls'])] = d['concept_cls']
    new_dataset['concept_cls'] = torch.from_numpy(concept_cls)
    max_gen_len = 1
    if 'gen' in dataset[0]:
        max_gen_len = max([len(d['gen']) for d in dataset]) + 1
        gen_out_seqs = np.full((batch_size, max_gen_len), -100, dtype=np.int64)
        gen_input_seqs = np.full((batch_size, max_gen_len), _EOS, dtype=np.
            int64)
        for i, d in enumerate(dataset):
            gen_input_seqs[i, 1:len(d['gen']) + 1] = d['gen']
            gen_out_seqs[i, :len(d['gen'])] = d['gen']
        gen_input_seqs[:, 0] = _BOS
        new_dataset['labels'] = torch.from_numpy(gen_out_seqs)
        new_dataset['decoder_input_ids'] = torch.from_numpy(gen_input_seqs)
        new_dataset['decoder_input_mask'] = (torch.from_numpy(
            gen_input_seqs) != _EOS).bool()
    copy_mention_flag = np.zeros((batch_size, max_gen_len, 5), dtype=np.int64)
    for i, d in enumerate(dataset):
        copy_mention_flag[i, :d['copy_mention_flag'].shape[0]] = d[
            'copy_mention_flag']
    new_dataset['copy_mention_flag'] = torch.from_numpy(copy_mention_flag)
    decoder_mention_flag = np.zeros((batch_size, max_gen_len,
        max_concept_len), dtype=np.int64)
    for i, d in enumerate(dataset):
        decoder_mention_flag[i, :d['decoder_mention_flag'].shape[0], :d[
            'decoder_mention_flag'].shape[1]] = d['decoder_mention_flag']
    new_dataset['decoder_mention_flag'] = torch.from_numpy(decoder_mention_flag
        )
    return new_dataset
