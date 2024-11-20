def verbose(self):
    """Returns a log string for each task in the results, detailing detection and classification outcomes."""
    log_string = ''
    probs = self.probs
    boxes = self.boxes
    if len(self) == 0:
        return (log_string if probs is not None else
            f'{log_string}(no detections), ')
    if probs is not None:
        log_string += (
            f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
            )
    if boxes:
        for c in boxes.cls.unique():
            n = (boxes.cls == c).sum()
            log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
    return log_string
