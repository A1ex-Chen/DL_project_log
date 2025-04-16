def clear_metrics(self):
    self.det_net_acc.clear()
    self.det_net_metrics.clear()
    self.det_net_cls_loss.clear()
    self.det_net_loc_loss.clear()
    self.det_net_total_loss.clear()
