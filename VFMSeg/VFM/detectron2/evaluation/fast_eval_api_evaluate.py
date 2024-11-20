def evaluate(self):
    """
        Run per image evaluation on given images and store results in self.evalImgs_cpp, a
        datastructure that isn't readable from Python but is used by a c++ implementation of
        accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
        self.evalImgs because this datastructure is a computational bottleneck.
        :return: None
        """
    tic = time.time()
    p = self.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
    logger.info('Evaluate annotation type *{}*'.format(p.iouType))
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
    maxDet = p.maxDets[-1]

    def convert_instances_to_cpp(instances, is_det=False):
        instances_cpp = []
        for instance in instances:
            instance_cpp = _C.InstanceAnnotation(int(instance['id']), 
                instance['score'] if is_det else instance.get('score', 0.0),
                instance['area'], bool(instance.get('iscrowd', 0)), bool(
                instance.get('ignore', 0)))
            instances_cpp.append(instance_cpp)
        return instances_cpp
    ground_truth_instances = [[convert_instances_to_cpp(self._gts[imgId,
        catId]) for catId in p.catIds] for imgId in p.imgIds]
    detected_instances = [[convert_instances_to_cpp(self._dts[imgId, catId],
        is_det=True) for catId in p.catIds] for imgId in p.imgIds]
    ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds
        ]
    if not p.useCats:
        ground_truth_instances = [[[o for c in i for o in c]] for i in
            ground_truth_instances]
        detected_instances = [[[o for c in i for o in c]] for i in
            detected_instances]
    self._evalImgs_cpp = _C.COCOevalEvaluateImages(p.areaRng, maxDet, p.
        iouThrs, ious, ground_truth_instances, detected_instances)
    self._evalImgs = None
    self._paramsEval = copy.deepcopy(self.params)
    toc = time.time()
    logger.info('COCOeval_opt.evaluate() finished in {:0.2f} seconds.'.
        format(toc - tic))
