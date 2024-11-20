def loadRes(self, detection_results, include_mask, is_image_mask=False):
    """Load result file and return a result api object.

    Args:
      detection_results: a dictionary containing predictions results.
      include_mask: a boolean, whether to include mask in detection results.
      is_image_mask: a boolean, where the predict mask is a whole image mask.

    Returns:
      res: result MaskCOCO api object
    """
    res = MaskCOCO()
    res.dataset['images'] = [img for img in self.dataset['images']]
    logging.info('Loading and preparing results...')
    predictions = self.load_predictions(detection_results, include_mask=
        include_mask, is_image_mask=is_image_mask)
    assert isinstance(predictions, list), 'results in not an array of objects'
    if predictions:
        image_ids = [pred['image_id'] for pred in predictions]
        assert set(image_ids) == set(image_ids) & set(self.getImgIds()
            ), 'Results do not correspond to current coco set'
        if predictions and 'bbox' in predictions[0] and predictions[0]['bbox']:
            res.dataset['categories'] = copy.deepcopy(self.dataset[
                'categories'])
            for idx, pred in enumerate(predictions):
                bb = pred['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if 'segmentation' not in pred:
                    pred['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                pred['area'] = bb[2] * bb[3]
                pred['id'] = idx + 1
                pred['iscrowd'] = 0
        elif 'segmentation' in predictions[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset[
                'categories'])
            for idx, pred in enumerate(predictions):
                pred['area'] = maskUtils.area(pred['segmentation'])
                if 'bbox' not in pred:
                    pred['bbox'] = maskUtils.toBbox(pred['segmentation'])
                pred['id'] = idx + 1
                pred['iscrowd'] = 0
        res.dataset['annotations'] = predictions
    res.createIndex()
    return res
