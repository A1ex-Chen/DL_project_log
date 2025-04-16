@property
def overall_iou(self):
    class_iou = np.array(self.class_iou.copy())
    class_iou[np.isnan(class_iou)] = 0
    return np.mean(class_iou)
