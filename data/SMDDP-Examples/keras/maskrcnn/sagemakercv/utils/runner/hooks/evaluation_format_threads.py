def format_threads(self):
    imgIds = []
    box_predictions = []
    mask_predictions = []
    for a_thread in self.threads:
        imgIds.extend(a_thread.result()[0])
        box_predictions.extend(a_thread.result()[1])
        mask_predictions.extend(a_thread.result()[2])
    return imgIds, box_predictions, mask_predictions
