def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """ batch_pc: BxNx3 """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio
        drop_idx = np.where(np.random.random(batch_pc.shape[1]) <=
            dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]
    return batch_pc
