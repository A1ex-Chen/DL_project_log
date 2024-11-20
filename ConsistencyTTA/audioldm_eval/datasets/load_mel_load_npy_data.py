def load_npy_data(loader):
    new_train = []
    for mel, waveform, filename in tqdm(loader):
        batch = batch.float().numpy()
        new_train.append(batch.reshape(-1))
    new_train = np.array(new_train)
    return new_train
