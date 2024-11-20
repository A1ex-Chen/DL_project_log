def dummy_model(self):

    def model(sample, t, *args):
        return sample * t / (t + 1)
    return model
