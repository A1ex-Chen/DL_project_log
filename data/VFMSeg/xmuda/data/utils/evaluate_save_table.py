def save_table(self, filename):
    from tabulate import tabulate
    header = ('overall acc', 'overall iou') + self.class_names
    table = [[self.overall_acc, self.overall_iou] + self.class_iou]
    with open(filename, 'w') as f:
        f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt=
            '.5f', numalign=None, stralign=None))
