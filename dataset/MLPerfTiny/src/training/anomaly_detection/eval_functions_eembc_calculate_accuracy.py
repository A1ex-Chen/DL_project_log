def calculate_accuracy(y_pred, labels):
    y_pred_label = np.argmax(y_pred, axis=1)
    correct = np.sum(labels == y_pred_label)
    accuracy = 100 * correct / len(y_pred)
    print(f'Overall accuracy = {accuracy:2.1f}')
    return accuracy
