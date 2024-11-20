def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type,
    kpt_oks_sigmas=None, use_fast_impl=True, img_ids=None,
    max_dets_per_image=None, refcoco=False):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0
    if iou_type == 'segm':
        coco_results = copy.deepcopy(coco_results)
        for c in coco_results:
            c.pop('bbox', None)
    coco_dt = coco_gt.loadRes(coco_results)
    if refcoco:
        coco_eval = RefCOCOeval(coco_gt, coco_dt, iou_type)
    else:
        coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt,
            coco_dt, iou_type)
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]
    else:
        assert len(max_dets_per_image
            ) >= 3, 'COCOeval requires maxDets (and max_dets_per_image) to have length at least 3'
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type != 'keypoints':
        coco_eval.params.maxDets = max_dets_per_image
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    if iou_type == 'keypoints':
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, 'kpt_oks_sigmas'
                ), 'pycocotools is too old!'
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        num_keypoints_dt = len(coco_results[0]['keypoints']) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))['keypoints']
            ) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, f'[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. Ground truth contains {num_keypoints_gt} keypoints. The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. They have to agree with each other. For meaning of OKS, please refer to http://cocodataset.org/#keypoints-eval.'
    coco_eval.evaluate()
    if not refcoco:
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval
