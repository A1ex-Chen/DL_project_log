def read_thread(self):
    while not self.stopped:
        if not self.q.full():
            ret, frame = self._vc.read()
            if not ret:
                self.stopped = True
            self.q.put(frame)
