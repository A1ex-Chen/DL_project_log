def data_wrapper(config, dataset):
    new_dataset = {'image_ids': [int(d['image_id']) for d in dataset]}
    new_dataset['gt'] = [d['gt'] for d in dataset]
    encoder_input_ids, encoder_mask = process_tensor([d['encoder_input_ids'
        ] for d in dataset], 0, output_mask=True)
    encoder_img_mask = process_tensor([d['encoder_img_mask'] for d in
        dataset], 0)
    encoder_cls = process_tensor([d['encoder_cls'] for d in dataset], 0)
    obj_feature = process_tensor([d['obj_feature_np'] for d in dataset], 2048)
    obj_box = process_tensor([d['obj_box_np'] for d in dataset], 8)
    new_dataset['encoder_input_ids'] = encoder_input_ids
    new_dataset['encoder_mask'] = encoder_mask
    new_dataset['encoder_img_mask'] = encoder_img_mask
    new_dataset['encoder_obj_feature'] = obj_feature
    new_dataset['encoder_obj_box'] = obj_box
    new_dataset['encoder_cls'] = encoder_cls
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
    encoder_rel_position = np.zeros((batch_size, max_encoder_len,
        max_encoder_len), dtype=np.int64)
    for i, d in enumerate(dataset):
        encoder_rel_position[i, :d['encoder_rel_position'].shape[0], :d[
            'encoder_rel_position'].shape[1]] = d['encoder_rel_position']
    new_dataset['encoder_rel_position'] = torch.from_numpy(encoder_rel_position
        )
    return new_dataset
