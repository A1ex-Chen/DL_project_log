def generate_queue(self):
    self.queue = []
    while len(self.queue) < self.total_size:
        class_set = [*range(self.classes_num)]
        random.shuffle(class_set)
        self.queue += [self.ipc[d][random.randint(0, len(self.ipc[d]) - 1)] for
            d in class_set]
    self.queue = self.queue[:self.total_size]
    logging.info('queue regenerated:%s' % self.queue[-5:])
