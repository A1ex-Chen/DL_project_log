def evaluateImg(self, imgId, catId, aRng, maxDet):
    """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return None
    for g in gt:
        if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([(-d['score']) for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]
        ) > 0 else self.ious[imgId, catId]
    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    if ious[dind, gind] < iou:
                        continue
                    iou = ious[dind, gind]
                    m = gind
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']
    a = np.array([(d['area'] < aRng[0] or d['area'] > aRng[1]) for d in dt]
        ).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    return {'image_id': imgId, 'category_id': catId, 'aRng': aRng, 'maxDet':
        maxDet, 'dtIds': [d['id'] for d in dt], 'gtIds': [g['id'] for g in
        gt], 'dtMatches': dtm, 'gtMatches': gtm, 'dtScores': [d['score'] for
        d in dt], 'gtIgnore': gtIg, 'dtIgnore': dtIg}
