@property
def class_iou(self):
    iou_list = []
    for i in range(self.num_classes):
        tp = self.confusion_matrix[i, i]
        p = self.confusion_matrix[:, i].sum()
        g = self.confusion_matrix[i, :].sum()
        union = p + g - tp
        if union == 0:
            iou = float('nan')
        else:
            iou = tp / union
        iou_list.append(iou)
    return iou_list
