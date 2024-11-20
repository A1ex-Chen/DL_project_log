def evaluate_coco_predictions(annotations_file, iou_types, predictions,
    verbose=False):
    cocoGt = COCO(annotation_file=annotations_file, use_ext=True)
    stat_dict = dict()
    for iou in iou_types:
        with contextlib.redirect_stdout(None):
            cocoDt = cocoGt.loadRes(predictions[iou], use_ext=True)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=iou, use_ext=True)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stat_dict[iou] = build_output_dict(iou, cocoEval.stats, verbose)
    return stat_dict
