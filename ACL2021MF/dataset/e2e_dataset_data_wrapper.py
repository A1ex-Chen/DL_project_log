def data_wrapper(dataset):
    new_dataset = {'gt': [d['gt'] for d in dataset], 'gt_mr': [d['gt_mr'] for
        d in dataset], 'ins_id': [d['ins_id'] for d in dataset]}
    encoder_input_ids, encoder_mask = process_tensor([d['encoder_input_ids'
        ] for d in dataset], 0, output_mask=True)
    encoder_class = process_tensor([d['encoder_class'] for d in dataset], 0,
        output_mask=False)
    new_dataset['encoder_input_ids'] = encoder_input_ids
    new_dataset['encoder_mask'] = encoder_mask
    new_dataset['encoder_cls'] = encoder_class
    max_gen_len = 1
    if 'cap' in dataset[0]:
        cap_decoder_input_ids, cap_decoder_mask = process_tensor([d['cap'] for
            d in dataset], 0, output_mask=True)
        cap_decoder_input_ids[cap_decoder_mask == 0] = -100
        new_dataset['cap_decoder_input_ids'] = cap_decoder_input_ids
        max_gen_len = cap_decoder_input_ids.size(1)
    batch_size = len(dataset)
    max_encoder_len = encoder_input_ids.size(1)
    mention_flag = np.zeros((batch_size, max_gen_len, max_encoder_len),
        dtype=np.int64)
    for i, d in enumerate(dataset):
        mention_flag[i, :d['mention_flag'].shape[0], :d['mention_flag'].
            shape[1]] = d['mention_flag']
    new_dataset['mention_flag'] = torch.from_numpy(mention_flag)
    return new_dataset
