def flow(self, single=False):
    while 1:
        x_list, y = self.get_slice(self.batch_size, single=single)
        yield x_list, y
