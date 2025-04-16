def save_one_json(predn, jdict, path, class_map, pred_masks):
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({'image_id': image_id, 'category_id': class_map[int(p[
            5])], 'bbox': [round(x, 3) for x in b], 'score': round(p[4], 5),
            'segmentation': rles[i]})
