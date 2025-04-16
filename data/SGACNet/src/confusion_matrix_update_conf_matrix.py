def update_conf_matrix(self, ground_truth, prediction):
    sess = tf.compat.v1.Session()
    current_confusion_matrix = sess.run(self.cm_func, feed_dict={self.
        ph_cm_y_true: ground_truth, self.ph_cm_y_pred: prediction})
    if self.overall_confusion_matrix is not None:
        self.overall_confusion_matrix += current_confusion_matrix
    else:
        self.overall_confusion_matrix = current_confusion_matrix
