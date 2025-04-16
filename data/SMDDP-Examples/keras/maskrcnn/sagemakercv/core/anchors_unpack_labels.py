def unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = OrderedDict()
    count = 0
    for level in range(self.min_level, self.max_level + 1):
        feat_size0 = int(self.image_size[0] / 2 ** level)
        feat_size1 = int(self.image_size[1] / 2 ** level)
        steps = feat_size0 * feat_size1 * self.get_anchors_per_location()
        indices = tf.range(count, count + steps)
        count += steps
        labels_unpacked[level] = tf.reshape(tf.gather(labels, indices), [
            feat_size0, feat_size1, -1])
    return labels_unpacked
