def reset_conf_matrix(self):
    self.overall_confusion_matrix = np.zeros((self.n_classes, self.n_classes))
