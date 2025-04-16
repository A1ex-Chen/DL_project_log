def evaluate_accuracy(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    return {'accuracy': accuracy}
