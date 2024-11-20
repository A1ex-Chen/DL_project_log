def get_maps(self, nc):
    return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)
