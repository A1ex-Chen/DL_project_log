@property
def global_avg(self):
    return self.iou.mean().item()
