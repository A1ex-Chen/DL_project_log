def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        item = ds[img_idx]
        image_id = item.image_id
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = item.image.shape[1]
        img_dict['width'] = item.image.shape[2]
        dataset['images'].append(img_dict)
        bboxes = item.bboxes
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = item.classes.tolist()
        areas = item.bboxes[:, 2] * item.bboxes[:, 3]
        areas = areas.tolist()
        iscrowd = item.difficulties.tolist()
        masks = item.masks
        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            ann['segmentation'] = coco_mask.encode(masks[i].numpy().astype(
                np.uint8))
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
