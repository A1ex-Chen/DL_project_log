def release(self):
    if self.released:
        return False
    signal.signal(self.sig, self.original_handler)
    self.released = True
    return True
