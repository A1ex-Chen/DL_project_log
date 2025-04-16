def update_metrics(self, cls_loss, loc_loss, cls_preds, labels, sampled):
    batch_size = cls_preds.shape[0]
    num_class = self._num_class
    if not self._encode_background_as_zeros:
        num_class += 1
    cls_preds = cls_preds.view(batch_size, -1, num_class)
    det_net_acc = self.det_net_acc(labels, cls_preds, sampled).numpy()[0]
    prec, recall = self.det_net_metrics(labels, cls_preds, sampled)
    prec = prec.numpy()
    recall = recall.numpy()
    det_net_cls_loss = self.det_net_cls_loss(cls_loss).numpy()[0]
    det_net_loc_loss = self.det_net_loc_loss(loc_loss).numpy()[0]
    ret = {'cls_loss': float(det_net_cls_loss), 'cls_loss_rt': float(
        cls_loss.data.cpu().numpy()), 'loc_loss': float(det_net_loc_loss),
        'loc_loss_rt': float(loc_loss.data.cpu().numpy()), 'det_net_acc':
        float(det_net_acc)}
    for i, thresh in enumerate(self.det_net_metrics.thresholds):
        ret[f'prec@{int(thresh * 100)}'] = float(prec[i])
        ret[f'rec@{int(thresh * 100)}'] = float(recall[i])
    return ret
