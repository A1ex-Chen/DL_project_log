def __next__(self):
    ret, frame = self.read()
    if ret:
        return frame
    raise StopIteration
