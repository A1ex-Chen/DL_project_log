def _summarizeKps():
    stats = np.zeros((10,))
    stats[0] = _summarize(1, maxDets=20)
    stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
    stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
    stats[3] = _summarize(1, maxDets=20, areaRng='medium')
    stats[4] = _summarize(1, maxDets=20, areaRng='large')
    stats[5] = _summarize(0, maxDets=20)
    stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
    stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
    stats[8] = _summarize(0, maxDets=20, areaRng='medium')
    stats[9] = _summarize(0, maxDets=20, areaRng='large')
    return stats
