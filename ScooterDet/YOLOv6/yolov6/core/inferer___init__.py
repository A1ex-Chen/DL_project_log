def __init__(self, nsamples: int=50):
    self.framerate = deque(maxlen=nsamples)
