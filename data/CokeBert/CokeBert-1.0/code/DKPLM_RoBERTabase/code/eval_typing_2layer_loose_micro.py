def loose_micro(true, pred):
    num_predicted_labels = 0.0
    num_true_labels = 0.0
    num_correct_labels = 0.0
    for true_labels, predicted_labels in zip(true, pred):
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(
            true_labels)))
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)
