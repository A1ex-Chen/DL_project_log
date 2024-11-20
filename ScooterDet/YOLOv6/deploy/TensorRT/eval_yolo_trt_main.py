def main():
    args = parse_args()
    check_args(args)
    if args.model.endswith('.onnx'):
        from onnx_to_trt import build_engine_from_onnx
        engine = build_engine_from_onnx(args.model, 'fp32', False)
        args.model = args.model.replace('.onnx', '.trt')
        with open(args.model, 'wb') as f:
            f.write(engine.serialize())
        print('Serialized the TensorRT engine to file: %s' % args.model)
    model_prefix = args.model.replace('.trt', '').split('/')[-1]
    results_file = 'results_{}.json'.format(model_prefix)
    if args.is_coco:
        data_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        model_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
    else:
        data_class = list(range(0, args.class_num))
        model_names = list(range(0, args.class_num))
    processor = Processor(model=args.model, is_end2end=args.is_end2end)
    image_names = [p for p in os.listdir(args.imgs_dir) if p.split('.')[-1]
        .lower() in IMG_FORMATS]
    with open(args.annotations) as f:
        coco_format_annotation = json.load(f)
    coco_format_imgs = [x['file_name'] for x in coco_format_annotation[
        'images']]
    imgname2id = {}
    for item in coco_format_annotation['images']:
        imgname2id[item['file_name']] = item['id']
    valid_images = []
    for img_name in image_names:
        img_name_wo_ext = os.path.splitext(img_name)[0]
        label_path = os.path.join(args.labels_dir, img_name_wo_ext + '.txt')
        if os.path.exists(label_path) and img_name in coco_format_imgs:
            valid_images.append(img_name)
        else:
            continue
    assert len(valid_images
        ) > 0, 'No valid images are found. Please check you image format or whether annotation file is match.'
    stats, seen = generate_results(data_class, model_names, args.
        do_pr_metric, args.plot_confusion_matrix, processor, args.imgs_dir,
        args.labels_dir, valid_images, results_file, args.conf_thres, args.
        iou_thres, args.is_coco, batch_size=args.batch_size, img_size=args.
        img_size, shrink_size=args.shrink_size, visualize=args.visualize,
        num_imgs_to_visualize=args.num_imgs_to_visualize, imgname2id=imgname2id
        )
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    if args.do_pr_metric:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            from yolov6.utils.metrics import ap_per_class
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=args.
                plot_curve, save_dir=args.save_dir, names=model_names)
            AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
            LOGGER.info(
                f'IOU 50 best mF1 thershold near {AP50_F1_max_idx / 1000.0}.')
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:,
                AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=len(
                model_names))
            s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels',
                'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
            LOGGER.info(s)
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5
            LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[
                AP50_F1_max_idx], map50, map))
            pr_metric_result = map50, map
            print('pr_metric results:', pr_metric_result)
            if args.verbose and len(model_names) > 1:
                for i, c in enumerate(ap_class):
                    LOGGER.info(pf % (model_names[c], seen, nt[c], p[i,
                        AP50_F1_max_idx], r[i, AP50_F1_max_idx], f1[i,
                        AP50_F1_max_idx], ap50[i], ap[i]))
            if args.plot_confusion_matrix:
                confusion_matrix.plot(save_dir=args.save_dir, names=list(
                    model_names))
        else:
            LOGGER.info('Calculate metric failed, might check dataset.')
            pr_metric_result = 0.0, 0.0
