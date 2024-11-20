def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    p = self.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.
            format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p
    self._prepare()
    catIds = p.catIds if p.useCats else [-1]
    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {(imgId, catId): computeIoU(imgId, catId) for imgId in p.
        imgIds for catId in catIds}
    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet) for catId in
        catIds for areaRng in p.areaRng for imgId in p.imgIds]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng),
        len(p.imgIds))
    self.evalImgs = evalImgs
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs
