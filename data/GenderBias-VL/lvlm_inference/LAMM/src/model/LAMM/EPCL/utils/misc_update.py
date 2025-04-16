def update(self, value, n=1):
    self.deque.append(value)
    self.count += n
    self.total += value * n
