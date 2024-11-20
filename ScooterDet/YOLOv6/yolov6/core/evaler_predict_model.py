def predict_model(self, model, dataloader, task):
    """Model prediction
        Predicts the whole dataset and gets the prediced results and inference time.
        """
    self.speed_result = torch.zeros(4, device=self.device)
    pred_results = []
    pbar = tqdm(dataloader, desc=f'Inferencing model in {task} datasets.',
        ncols=NCOLS)
    if self.do_pr_metric:
        stats, ap = [], []
        seen = 0
        iouv = torch.linspace(0.5, 0.95, 10)
        niou = iouv.numel()
        if self.plot_confusion_matrix:
            from yolov6.utils.metrics import ConfusionMatrix
            confusion_matrix = ConfusionMatrix(nc=model.nc)
    for i, (imgs, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        imgs = imgs.to(self.device, non_blocking=True)
        imgs = imgs.half() if self.half else imgs.float()
        imgs /= 255
        self.speed_result[1] += time_sync() - t1
        t2 = time_sync()
        outputs, _ = model(imgs)
        self.speed_result[2] += time_sync() - t2
        t3 = time_sync()
        outputs = non_max_suppression(outputs, self.conf_thres, self.
            iou_thres, multi_label=True)
        self.speed_result[3] += time_sync() - t3
        self.speed_result[0] += len(outputs)
        if self.do_pr_metric:
            import copy
            eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])
        pred_results.extend(self.convert_to_coco_format(outputs, imgs,
            paths, shapes, self.ids))
        if i == 0:
            vis_num = min(len(imgs), 8)
            vis_outputs = outputs[:vis_num]
            vis_paths = paths[:vis_num]
        if not self.do_pr_metric:
            continue
        for si, pred in enumerate(eval_outputs):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                        torch.Tensor(), torch.Tensor(), tcls))
                continue
            predn = pred.clone()
            self.scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][
                0], shapes[si][1])
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            if nl:
                from yolov6.utils.nms import xywh2xyxy
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= imgs[si].shape[1:][1]
                tbox[:, [1, 3]] *= imgs[si].shape[1:][0]
                self.scale_coords(imgs[si].shape[1:], tbox, shapes[si][0],
                    shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                from yolov6.utils.metrics import process_batch
                correct = process_batch(predn, labelsn, iouv)
                if self.plot_confusion_matrix:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(),
                tcls))
    if self.do_pr_metric:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            from yolov6.utils.metrics import ap_per_class
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.
                plot_curve, save_dir=self.save_dir, names=model.names)
            AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
            LOGGER.info(
                f'IOU 50 best mF1 thershold near {AP50_F1_max_idx / 1000.0}.')
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:,
                AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=model.nc)
            s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels',
                'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
            LOGGER.info(s)
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5
            LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[
                AP50_F1_max_idx], map50, map))
            self.pr_metric_result = map50, map
            if self.verbose and model.nc > 1:
                for i, c in enumerate(ap_class):
                    LOGGER.info(pf % (model.names[c], seen, nt[c], p[i,
                        AP50_F1_max_idx], r[i, AP50_F1_max_idx], f1[i,
                        AP50_F1_max_idx], ap50[i], ap[i]))
            if self.plot_confusion_matrix:
                confusion_matrix.plot(save_dir=self.save_dir, names=list(
                    model.names))
        else:
            LOGGER.info('Calculate metric failed, might check dataset.')
            self.pr_metric_result = 0.0, 0.0
    return pred_results, vis_outputs, vis_paths
