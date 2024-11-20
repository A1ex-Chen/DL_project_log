def evaluate(self):
    """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
    tic = time.time()
    print('Running per image evaluation...')
    p = self.params
    if not p.useSegm is None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.
            format(p.iouType))
    print('Evaluate annotation type *{}*'.format(p.iouType))
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
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))
