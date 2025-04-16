def computeIoU(self, imgId, catId):
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return []
    inds = np.argsort([(-d['score']) for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    if len(dt) > p.maxDets[-1]:
        dt = dt[0:p.maxDets[-1]]
    assert p.iouType == 'bbox', 'unsupported iouType for iou computation'
    g = [g['bbox'] for g in gt]
    d = [d['bbox'] for d in dt]
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = self.compute_iou_dt_gt(d, g, iscrowd)
    return ious
