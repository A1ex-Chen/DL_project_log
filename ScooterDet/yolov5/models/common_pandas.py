def pandas(self):
    new = copy(self)
    ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
    cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'
    for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
        a = [[(x[:5] + [int(x[5]), self.names[int(x[5])]]) for x in x.
            tolist()] for x in getattr(self, k)]
        setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
    return new
