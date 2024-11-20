def move_stopping_criteria_device(self, device, dtype=torch.float32):
    self.stop_words_ids = [stop_tensor.to(device, dtype=dtype) for
        stop_tensor in self.stop_words_ids]
    self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        stops=self.stop_words_ids)])
