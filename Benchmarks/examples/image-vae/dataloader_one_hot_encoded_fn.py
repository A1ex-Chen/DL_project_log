def one_hot_encoded_fn(self, row):
    return np.array(map(lambda x: self.one_hot_array(x, self.vocab)), self.
        one_hot_index(row, self.vocab))
