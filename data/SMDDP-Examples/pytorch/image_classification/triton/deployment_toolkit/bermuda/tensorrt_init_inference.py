def init_inference(self, model: Model):
    return TensorRTRunnerSession(model=model)
