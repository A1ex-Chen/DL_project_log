def update(self, predictions=None):
    for obj in predictions:
        new_row = [self.frame_number, obj.id, obj.estimate[0, 0], obj.
            estimate[0, 1], obj.estimate[1, 0] - obj.estimate[0, 0], obj.
            estimate[1, 1] - obj.estimate[0, 1], -1, -1, -1, -1]
        if np.shape(self.matrix_predictions)[0] == 0:
            self.matrix_predictions = new_row
        else:
            self.matrix_predictions = np.vstack((self.matrix_predictions,
                new_row))
    self.frame_number += 1
    try:
        next(self.progress_bar_iter)
    except StopIteration:
        self.matrixes_predictions.append(self.matrix_predictions)
        return
