def evaluate(self):
    comm.synchronize()
    if comm.get_rank() > 0:
        return
    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval
    self._logger.info('Evaluating results under {} ...'.format(self._temp_dir))
    cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
    cityscapes_eval.args.predictionWalk = None
    cityscapes_eval.args.JSONOutput = False
    cityscapes_eval.args.colorized = False
    gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
    groundTruthImgList = glob.glob(os.path.join(gt_dir, '*',
        '*_gtFine_labelIds.png'))
    assert len(groundTruthImgList
        ), 'Cannot find any ground truth images to use for evaluation. Searched for: {}'.format(
        cityscapes_eval.args.groundTruthSearch)
    predictionImgList = []
    for gt in groundTruthImgList:
        predictionImgList.append(cityscapes_eval.getPrediction(
            cityscapes_eval.args, gt))
    results = cityscapes_eval.evaluateImgLists(predictionImgList,
        groundTruthImgList, cityscapes_eval.args)
    ret = OrderedDict()
    ret['sem_seg'] = {'IoU': 100.0 * results['averageScoreClasses'], 'iIoU':
        100.0 * results['averageScoreInstClasses'], 'IoU_sup': 100.0 *
        results['averageScoreCategories'], 'iIoU_sup': 100.0 * results[
        'averageScoreInstCategories']}
    self._working_dir.cleanup()
    return ret
