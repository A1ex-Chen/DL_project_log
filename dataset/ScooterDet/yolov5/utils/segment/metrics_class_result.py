def class_result(self, i):
    return self.metric_box.class_result(i) + self.metric_mask.class_result(i)
