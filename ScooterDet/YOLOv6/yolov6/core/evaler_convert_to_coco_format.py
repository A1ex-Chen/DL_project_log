def convert_to_coco_format(self, outputs, imgs, paths, shapes, ids):
    pred_results = []
    for i, pred in enumerate(outputs):
        if len(pred) == 0:
            continue
        path, shape = Path(paths[i]), shapes[i][0]
        self.scale_coords(imgs[i].shape[1:], pred[:, :4], shape, shapes[i][1])
        image_id = int(path.stem) if self.is_coco else path.stem
        bboxes = self.box_convert(pred[:, 0:4])
        bboxes[:, :2] -= bboxes[:, 2:] / 2
        cls = pred[:, 5]
        scores = pred[:, 4]
        for ind in range(pred.shape[0]):
            category_id = ids[int(cls[ind])]
            bbox = [round(x, 3) for x in bboxes[ind].tolist()]
            score = round(scores[ind].item(), 5)
            pred_data = {'image_id': image_id, 'category_id': category_id,
                'bbox': bbox, 'score': score}
            pred_results.append(pred_data)
    return pred_results
