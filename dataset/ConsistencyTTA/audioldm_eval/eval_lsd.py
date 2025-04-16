def lsd(self, audio1, audio2):
    result = self.lsd_metric.evaluation(audio1, audio2, None)
    return result
