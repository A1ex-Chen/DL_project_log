def calculate_all_accuracies(y_pred, labels, classes):
    n_classes = len(classes)
    accuracies = np.zeros(n_classes)
    for class_item in range(n_classes):
        true_positives = 0
        for i in range(len(y_pred)):
            if labels[i] == class_item:
                y_pred_label = np.argmax(y_pred[i, :])
                if labels[i] == y_pred_label:
                    true_positives += 1
        accuracies[class_item] = 100 * true_positives / np.sum(labels ==
            class_item)
        print(
            f'Accuracy = {accuracies[class_item]:2.1f} ({classes[class_item]})'
            )
    return accuracies
