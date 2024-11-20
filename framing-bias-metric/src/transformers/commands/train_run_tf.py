def run_tf(self):
    self.pipeline.fit(self.train_dataset, validation_data=self.
        valid_dataset, validation_split=self.validation_split,
        learning_rate=self.learning_rate, adam_epsilon=self.adam_epsilon,
        train_batch_size=self.train_batch_size, valid_batch_size=self.
        valid_batch_size)
    self.pipeline.save_pretrained(self.output)
