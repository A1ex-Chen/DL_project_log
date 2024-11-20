@property
def class_seg_acc(self):
    return [(self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
        ) for i in range(self.num_classes)]
