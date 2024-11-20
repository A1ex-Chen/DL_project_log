def collate_fn(batch):
    """
    Collate function for wdsdataloader.
    batch: a list of dict, each dict is a sample
    """
    batch_dict = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], dict):
            batch_dict[k] = {}
            for kk in batch[0][k].keys():
                tmp = []
                for i in range(len(batch)):
                    tmp.append(batch[i][k][kk])
                batch_dict[k][kk] = torch.vstack(tmp)
        elif isinstance(batch[0][k], torch.Tensor):
            batch_dict[k] = torch.stack([sample[k] for sample in batch])
        elif isinstance(batch[0][k], np.ndarray):
            batch_dict[k] = torch.tensor(np.stack([sample[k] for sample in
                batch]))
        else:
            batch_dict[k] = [sample[k] for sample in batch]
    return batch_dict
