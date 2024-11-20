@property
def avg_forward_time(self):
    return self._total_forward_time / self._total_inference_count
