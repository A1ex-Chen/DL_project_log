def on_test_begin(self, logs: MutableMapping[Text, Any]=None):
    self.model.optimizer.swap_weights()
