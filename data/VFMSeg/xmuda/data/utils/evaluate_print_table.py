def print_table(self):
    from tabulate import tabulate
    header = ['Class', 'Accuracy', 'IOU', 'Total']
    seg_acc_per_class = self.class_seg_acc
    iou_per_class = self.class_iou
    table = []
    for ind, class_name in enumerate(self.class_names):
        table.append([class_name, seg_acc_per_class[ind] * 100, 
            iou_per_class[ind] * 100, int(self.confusion_matrix[ind].sum())])
    return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')
