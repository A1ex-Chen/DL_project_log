def convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, ids
    ):
    pred_results = []
    for i, (num, detbox, detscore, detcls) in enumerate(zip(nums, boxes,
        scores, classes)):
        n = int(num[0])
        if n == 0:
            continue
        path, shape = Path(paths[i]), shapes[i][0]
        gain = shapes[i][1][0][0]
        pad = torch.tensor(shapes[i][1][1] * 2).to(self.device)
        detbox = detbox[:n, :]
        detbox -= pad
        detbox /= gain
        detbox[:, 0].clamp_(0, shape[1])
        detbox[:, 1].clamp_(0, shape[0])
        detbox[:, 2].clamp_(0, shape[1])
        detbox[:, 3].clamp_(0, shape[0])
        detbox[:, 2:] = detbox[:, 2:] - detbox[:, :2]
        detscore = detscore[:n]
        detcls = detcls[:n]
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        for ind in range(n):
            category_id = ids[int(detcls[ind])]
            bbox = [round(x, 3) for x in detbox[ind].tolist()]
            score = round(detscore[ind].item(), 5)
            pred_data = {'image_id': image_id, 'category_id': category_id,
                'bbox': bbox, 'score': score}
            pred_results.append(pred_data)
    return pred_results
