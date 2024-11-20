def build_confusion_matrix_graph(self):
    tf.compat.v1.disable_eager_execution()
    self.ph_cm_y_true = tf.compat.v1.placeholder(dtype=self.dtype, shape=None)
    self.ph_cm_y_pred = tf.compat.v1.placeholder(dtype=self.dtype, shape=None)
    return tf.math.confusion_matrix(labels=self.ph_cm_y_true, predictions=
        self.ph_cm_y_pred, num_classes=self.n_classes)
