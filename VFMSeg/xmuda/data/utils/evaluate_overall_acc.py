@property
def overall_acc(self):
    return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.
        confusion_matrix)
