@property
def avg_postprocess_time(self):
    return self._total_postprocess_time / self._total_inference_count
