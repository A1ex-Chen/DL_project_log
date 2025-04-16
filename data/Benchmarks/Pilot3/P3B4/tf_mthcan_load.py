def load(self, filename):
    self.saver.restore(self.sess, filename)
