def read_targets(self, data_file, partition):
    """Get dictionary of targets specified by user."""
    if partition == 'train':
        label = 'y_train'
    else:
        label = 'y_val'
    tasks = {'response': torch.tensor(pd.read_hdf(data_file, label)['AUC'].
        apply(lambda x: 1 if x < 0.5 else 0))}
    return tasks
