def get_image_and_label(self, index):
    """Get and return label information from the dataset."""
    label = deepcopy(self.labels[index])
    label.pop('shape', None)
    label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(
        index)
    label['ratio_pad'] = label['resized_shape'][0] / label['ori_shape'][0
        ], label['resized_shape'][1] / label['ori_shape'][1]
    if self.rect:
        label['rect_shape'] = self.batch_shapes[self.batch[index]]
    return self.update_labels_info(label)
