def __init__(self, data, partition='train', fold=0, source=None, batch_size
    =32, shuffle=True, single=False, rank=0, total_ranks=1):
    self.data = data
    self.partition = partition
    self.batch_size = batch_size
    self.single = single
    if partition == 'train':
        index = data.train_indexes[fold]
    elif partition == 'val':
        index = data.val_indexes[fold]
    else:
        index = data.test_indexes[fold] if hasattr(data, 'test_indexes') else [
            ]
    if source:
        df = data.df_response[['Source']].iloc[index, :]
        index = df.index[df['Source'] == source]
    if shuffle:
        index = np.random.permutation(index)
    samples_per_rank = len(index) // total_ranks
    samples_per_rank = self.batch_size * (samples_per_rank // self.batch_size)
    self.index = index[rank * samples_per_rank:(rank + 1) * samples_per_rank]
    self.index_cycle = cycle(self.index)
    self.size = len(self.index)
    self.steps = self.size // self.batch_size
    print(
        'partition:{0}, rank:{1}, sharded index size:{2}, batch_size:{3}, steps:{4}'
        .format(partition, rank, self.size, self.batch_size, self.steps))
