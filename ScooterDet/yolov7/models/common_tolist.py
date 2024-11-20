def tolist(self):
    x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for
        i in range(self.n)]
    for d in x:
        for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
            setattr(d, k, getattr(d, k)[0])
    return x
