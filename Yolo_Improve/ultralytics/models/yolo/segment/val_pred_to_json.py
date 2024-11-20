def pred_to_json(self, predn, filename, pred_masks):
    """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
    from pycocotools.mask import encode

    def single_encode(x):
        """Encode predicted masks as RLE and append results to jdict."""
        rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    stem = Path(filename).stem
    image_id = int(stem) if stem.isnumeric() else stem
    box = ops.xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        self.jdict.append({'image_id': image_id, 'category_id': self.
            class_map[int(p[5])], 'bbox': [round(x, 3) for x in b], 'score':
            round(p[4], 5), 'segmentation': rles[i]})
