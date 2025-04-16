def loose_macro(true, pred):
    num_entities = len(true)
    p = 0.0
    r = 0.0
    for true_labels, predicted_labels in zip(true, pred):
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))
                ) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))
                ) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1(precision, recall)
