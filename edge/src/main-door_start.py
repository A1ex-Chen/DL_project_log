def start(self):
    Thread(target=self.update, args=()).start()
    return self
