def set_seed(self):
    random.seed(self.seed)
    tf.random.set_seed(self.seed)
