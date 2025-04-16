def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
