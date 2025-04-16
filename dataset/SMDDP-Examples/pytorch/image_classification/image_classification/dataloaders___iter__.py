def __iter__(self):
    while True:
        yield self.input_data, self.input_target
