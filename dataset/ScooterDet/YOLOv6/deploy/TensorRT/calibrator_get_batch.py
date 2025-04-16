def get_batch(self, names):
    print('######################')
    print(names)
    print('######################')
    batch = self.stream.next_batch()
    if not batch.size:
        return None
    cuda.memcpy_htod(self.d_input, batch)
    return [int(self.d_input)]
