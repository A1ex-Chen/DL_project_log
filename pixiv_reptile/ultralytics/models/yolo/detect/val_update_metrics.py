def update_metrics(self, preds, batch):
    """Metrics."""
    for si, pred in enumerate(preds):
        self.seen += 1
        npr = len(pred)
        stat = dict(conf=torch.zeros(0, device=self.device), pred_cls=torch
            .zeros(0, device=self.device), tp=torch.zeros(npr, self.niou,
            dtype=torch.bool, device=self.device))
        pbatch = self._prepare_batch(si, batch)
        cls, bbox = pbatch.pop('cls'), pbatch.pop('bbox')
        nl = len(cls)
        stat['target_cls'] = cls
        stat['target_img'] = cls.unique()
        if npr == 0:
            if nl:
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])
                if self.args.plots:
                    self.confusion_matrix.process_batch(detections=None,
                        gt_bboxes=bbox, gt_cls=cls)
            continue
        if self.args.single_cls:
            pred[:, 5] = 0
        predn = self._prepare_pred(pred, pbatch)
        stat['conf'] = predn[:, 4]
        stat['pred_cls'] = predn[:, 5]
        if nl:
            stat['tp'] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
        for k in self.stats.keys():
            self.stats[k].append(stat[k])
        if self.args.save_json:
            self.pred_to_json(predn, batch['im_file'][si])
        if self.args.save_txt:
            file = (self.save_dir / 'labels' /
                f"{Path(batch['im_file'][si]).stem}.txt")
            self.save_one_txt(predn, self.args.save_conf, pbatch[
                'ori_shape'], file)
