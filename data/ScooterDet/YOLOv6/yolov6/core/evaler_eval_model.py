def eval_model(self, pred_results, model, dataloader, task):
    """Evaluate models
        For task speed, this function only evaluates the speed of model and outputs inference time.
        For task val, this function evaluates the speed and mAP by pycocotools, and returns
        inference time and mAP value.
        """
    LOGGER.info(f'\nEvaluating speed.')
    self.eval_speed(task)
    if not self.do_coco_metric and self.do_pr_metric:
        return self.pr_metric_result
    LOGGER.info(f'\nEvaluating mAP by pycocotools.')
    if task != 'speed' and len(pred_results):
        if 'anno_path' in self.data:
            anno_json = self.data['anno_path']
        else:
            task = 'val' if task == 'train' else task
            if not isinstance(self.data[task], list):
                self.data[task] = [self.data[task]]
            dataset_root = os.path.dirname(os.path.dirname(self.data[task][0]))
            base_name = os.path.basename(self.data[task][0])
            anno_json = os.path.join(dataset_root, 'annotations',
                f'instances_{base_name}.json')
        pred_json = os.path.join(self.save_dir, 'predictions.json')
        LOGGER.info(f'Saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(pred_results, f)
        anno = COCO(anno_json)
        pred = anno.loadRes(pred_json)
        cocoEval = COCOeval(anno, pred, 'bbox')
        if self.is_coco:
            imgIds = [int(os.path.basename(x).split('.')[0]) for x in
                dataloader.dataset.img_paths]
            cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        if self.verbose:
            import copy
            val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
            val_dataset_anns_count = 0
            label_count_dict = {'images': set(), 'anns': 0}
            label_count_dicts = [copy.deepcopy(label_count_dict) for _ in
                range(model.nc)]
            for _, ann_i in cocoEval.cocoGt.anns.items():
                if ann_i['ignore']:
                    continue
                val_dataset_anns_count += 1
                nc_i = self.coco80_to_coco91_class().index(ann_i['category_id']
                    ) if self.is_coco else ann_i['category_id']
                label_count_dicts[nc_i]['images'].add(ann_i['image_id'])
                label_count_dicts[nc_i]['anns'] += 1
            s = ('%-16s' + '%12s' * 7) % ('Class', 'Labeled_images',
                'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5',
                'mAP@.5:.95')
            LOGGER.info(s)
            coco_p = cocoEval.eval['precision']
            coco_p_all = coco_p[:, :, :, 0, 2]
            map = np.mean(coco_p_all[coco_p_all > -1])
            coco_p_iou50 = coco_p[0, :, :, 0, 2]
            map50 = np.mean(coco_p_iou50[coco_p_iou50 > -1])
            mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii] > -1]) for
                ii in range(coco_p_iou50.shape[0])])
            mr = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) + 
                1, endpoint=True)
            mf1 = 2 * mp * mr / (mp + mr + 1e-16)
            i = mf1.argmax()
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5
            LOGGER.info(pf % ('all', val_dataset_img_count,
                val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))
            for nc_i in range(model.nc):
                coco_p_c = coco_p[:, :, nc_i, 0, 2]
                map = np.mean(coco_p_c[coco_p_c > -1])
                coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
                map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50 > -1])
                p = coco_p_c_iou50
                r = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) +
                    1, endpoint=True)
                f1 = 2 * p * r / (p + r + 1e-16)
                i = f1.argmax()
                LOGGER.info(pf % (model.names[nc_i], len(label_count_dicts[
                    nc_i]['images']), label_count_dicts[nc_i]['anns'], p[i],
                    r[i], f1[i], map50, map))
        cocoEval.summarize()
        map, map50 = cocoEval.stats[:2]
        model.float()
        if task != 'train':
            LOGGER.info(f'Results saved to {self.save_dir}')
        return map50, map
    return 0.0, 0.0
