def _prepare(self):
    """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

    def _toMask(anns, coco):
        for ann in anns:
            rle = coco.annToRLE(ann)
            ann['segmentation'] = rle
    p = self.params
    if p.useCats:
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds,
            catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds,
            catIds=p.catIds))
    else:
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))
    if p.iouType == 'segm':
        _toMask(gts, self.cocoGt)
        _toMask(dts, self.cocoDt)
    for gt in gts:
        gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
        gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        if p.iouType == 'keypoints':
            gt['ignore'] = gt['num_keypoints'] == 0 or gt['ignore']
    self._gts = defaultdict(list)
    self._dts = defaultdict(list)
    for gt in gts:
        self._gts[gt['image_id'], gt['category_id']].append(gt)
    for dt in dts:
        self._dts[dt['image_id'], dt['category_id']].append(dt)
    self.evalImgs = defaultdict(list)
    self.eval = {}
