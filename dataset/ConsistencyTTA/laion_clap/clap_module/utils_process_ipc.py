def process_ipc(index_path, classes_num, filename):
    logging.info('Load Data...............')
    ipc = [[] for _ in range(classes_num)]
    with h5py.File(index_path, 'r') as f:
        for i in tqdm(range(len(f['target']))):
            t_class = np.where(f['target'][i])[0]
            for t in t_class:
                ipc[t].append(i)
    print(ipc)
    np.save(filename, ipc)
    logging.info('Load Data Succeed...............')
