def update_metrics(self, preds, batch):
    """Metrics."""
    for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
        self.seen += 1
        npr = len(pred)
        stat = dict(conf=torch.zeros(0, device=self.device), pred_cls=torch
            .zeros(0, device=self.device), tp=torch.zeros(npr, self.niou,
            dtype=torch.bool, device=self.device), tp_m=torch.zeros(npr,
            self.niou, dtype=torch.bool, device=self.device))
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
        gt_masks = pbatch.pop('masks')
        if self.args.single_cls:
            pred[:, 5] = 0
        predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
        stat['conf'] = predn[:, 4]
        stat['pred_cls'] = predn[:, 5]
        if nl:
            stat['tp'] = self._process_batch(predn, bbox, cls)
            stat['tp_m'] = self._process_batch(predn, bbox, cls, pred_masks,
                gt_masks, self.args.overlap_mask, masks=True)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
        for k in self.stats.keys():
            self.stats[k].append(stat[k])
        pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
        if self.args.plots and self.batch_i < 3:
            self.plot_masks.append(pred_masks[:15].cpu())
        if self.args.save_json:
            pred_masks = ops.scale_image(pred_masks.permute(1, 2, 0).
                contiguous().cpu().numpy(), pbatch['ori_shape'], ratio_pad=
                batch['ratio_pad'][si])
            self.pred_to_json(predn, batch['im_file'][si], pred_masks)
