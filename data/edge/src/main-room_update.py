def update(self):
    while True:
        if self.stopped:
            return
        self.grabbed, self.frame = self.stream.read()
