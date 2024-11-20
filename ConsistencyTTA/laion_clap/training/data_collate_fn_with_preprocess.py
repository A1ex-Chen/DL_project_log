def collate_fn_with_preprocess(batch, audio_ext, text_ext, max_len,
    audio_cfg, args):
    """
    Collate function for wdsdataloader.
    batch: a list of dict, each dict is a sample
    """
    class_index_dict = copy.deepcopy(args.class_index_dict)
    data_filling = args.data_filling
    data_truncating = args.data_truncating
    text_augment_selection = args.text_augment_selection
    tmodel = args.tmodel
    data_preprocessed = []
    for sample in batch:
        data_preprocessed.append(preprocess_single(sample, audio_ext,
            text_ext, max_len, audio_cfg, tmodel, class_index_dict,
            data_filling, data_truncating, text_augment_selection))
    batch_dict = {}
    for k in data_preprocessed[0].keys():
        if isinstance(data_preprocessed[0][k], dict):
            batch_dict[k] = {}
            for kk in data_preprocessed[0][k].keys():
                tmp = []
                for i in range(len(data_preprocessed)):
                    tmp.append(data_preprocessed[i][k][kk])
                batch_dict[k][kk] = torch.vstack(tmp)
        elif isinstance(data_preprocessed[0][k], torch.Tensor):
            batch_dict[k] = torch.stack([sample[k] for sample in
                data_preprocessed])
        elif isinstance(data_preprocessed[0][k], np.ndarray):
            batch_dict[k] = torch.tensor(np.stack([sample[k] for sample in
                data_preprocessed]))
        else:
            batch_dict[k] = [sample[k] for sample in data_preprocessed]
    del data_preprocessed
    return batch_dict
