def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.
            flow_generator)
    bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
    for i, j in enumerate(index_array):
        x = self.X[j]
        x = self.insertnoise(x, corruption_level=self.p)
        bX[i] = x
    bY = self.y[index_array]
    return bX, bY
