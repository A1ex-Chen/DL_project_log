def compute_RRM(origin_acc, crp_acc, dataset_name):
    rd_acc = rand_acc[dataset_name]['vanilla']
    return (crp_acc - rd_acc) / (origin_acc - rd_acc) * 100
