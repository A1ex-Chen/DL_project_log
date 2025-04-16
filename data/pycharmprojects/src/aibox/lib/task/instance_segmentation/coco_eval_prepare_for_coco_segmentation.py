def prepare_for_coco_segmentation(self, predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        scores = prediction['scores']
        labels = prediction['labels']
        masks = prediction['masks']
        masks = masks > 0.5
        scores = prediction['scores'].tolist()
        labels = prediction['labels'].tolist()
        rles = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=
            np.uint8, order='F'))[0] for mask in masks]
        for rle in rles:
            rle['counts'] = rle['counts'].decode('utf-8')
        coco_results.extend([{'image_id': original_id, 'category_id':
            labels[k], 'segmentation': rle, 'score': scores[k]} for k, rle in
            enumerate(rles)])
    return coco_results
