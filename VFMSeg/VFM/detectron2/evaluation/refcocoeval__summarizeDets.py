def _summarizeDets():
    stats = np.zeros((12,))
    stats[0] = _summarize(1)
    stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
    stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
    stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
    return stats
