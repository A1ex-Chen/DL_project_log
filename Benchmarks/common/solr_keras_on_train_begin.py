def on_train_begin(self, logs={}):
    """Start clock to calculate timeout"""
    self.run_timestamp = datetime.now()
