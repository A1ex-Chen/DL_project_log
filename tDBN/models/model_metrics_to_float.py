def metrics_to_float(self):
    self.det_net_acc.float()
    self.det_net_metrics.float()
    self.det_net_cls_loss.float()
    self.det_net_loc_loss.float()
    self.det_net_total_loss.float()
