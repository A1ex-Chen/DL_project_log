def model_generate(self, *args, **kwargs):
    with self.model.maybe_autocast():
        output = self.model.llama_model.generate(*args, **kwargs)
    return output
