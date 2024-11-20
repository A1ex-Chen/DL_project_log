def one_hot_array(self, i, n):
    return map(int, [(ix == i) for ix in range(n)])
