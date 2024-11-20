def generate_results(data_class, model_names, do_pr_metric,
    plot_confusion_matrix, processor, imgs_dir, labels_dir, valid_images,
    results_file, conf_thres, iou_thres, is_coco, batch_size=1, img_size=[
    640, 640], shrink_size=0, visualize=False, num_imgs_to_visualize=0,
    imgname2id={}):
    """Run detection on each jpg and write results to file."""
    results = []
    pbar = tqdm(range(math.ceil(len(valid_images) / batch_size)), desc=
        'TRT-Model test in val datasets.')
    idx = 0
    num_visualized = 0
    stats = []
    seen = 0
    if do_pr_metric:
        iouv = torch.linspace(0.5, 0.95, 10)
        niou = iouv.numel()
        if plot_confusion_matrix:
            from yolov6.utils.metrics import ConfusionMatrix
            confusion_matrix = ConfusionMatrix(nc=len(model_names))
    for _ in pbar:
        preprocessed_imgs = []
        source_imgs = []
        image_ids = []
        shapes = []
        targets = []
        for i in range(batch_size):
            if idx == len(valid_images):
                break
            img = cv2.imread(os.path.join(imgs_dir, valid_images[idx]))
            imgs_name = os.path.splitext(valid_images[idx])[0]
            label_path = os.path.join(labels_dir, imgs_name + '.txt')
            with open(label_path, 'r') as f:
                target = [x.split() for x in f.read().strip().splitlines() if
                    len(x)]
                target = np.array(target, dtype=np.float32)
                targets.append(target)
            img_src = img.copy()
            h0, w0 = img.shape[:2]
            r = (max(img_size) - shrink_size) / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                    )
            h, w = img.shape[:2]
            preprocessed_img, pad = processor.pre_process(img)
            preprocessed_imgs.append(preprocessed_img)
            source_imgs.append(img_src)
            shape = (h0, w0), ((h / h0, w / w0), pad)
            shapes.append(shape)
            assert valid_images[idx] in imgname2id.keys(
                ), f'valid_images[idx] not in annotations you provided.'
            image_ids.append(imgname2id[valid_images[idx]])
            idx += 1
        output = processor.inference(torch.stack(preprocessed_imgs, axis=0))
        for j in range(len(shapes)):
            pred = processor.post_process(output[j].unsqueeze(0), shapes[j],
                conf_thres=conf_thres, iou_thres=iou_thres)
            if visualize and num_visualized < num_imgs_to_visualize:
                image = source_imgs[i]
            for p in pred:
                x = float(p[0])
                y = float(p[1])
                w = float(p[2] - p[0])
                h = float(p[3] - p[1])
                s = float(p[4])
                results.append({'image_id': image_ids[j], 'category_id': 
                    data_class[int(p[5])] if is_coco else int(p[5]), 'bbox':
                    [round(x, 3) for x in [x, y, w, h]], 'score': round(s, 5)})
                if visualize and num_visualized < num_imgs_to_visualize:
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int
                        (y + h)), (255, 0, 0), 1)
            if do_pr_metric:
                import copy
                target = targets[j]
                labels = target.copy()
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                seen += 1
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool
                            ), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                predn = pred.clone()
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    from yolov6.utils.nms import xywh2xyxy
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= shapes[j][0][1]
                    tbox[:, [1, 3]] *= shapes[j][0][0]
                    labelsn = torch.cat((torch.from_numpy(labels[:, 0:1]).
                        cpu(), torch.from_numpy(tbox).cpu()), 1)
                    from yolov6.utils.metrics import process_batch
                    correct = process_batch(predn.cpu(), labelsn.cpu(), iouv)
                    if plot_confusion_matrix:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].
                    cpu(), tcls))
            if visualize and num_visualized < num_imgs_to_visualize:
                print('saving to %d.jpg' % num_visualized)
                err_code = cv2.imwrite('./%d.jpg' % num_visualized, image)
                num_visualized += 1
    with open(results_file, 'w') as f:
        LOGGER.info(f'saving coco format detection resuslt to {results_file}')
        f.write(json.dumps(results, indent=4))
    return stats, seen
