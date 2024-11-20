def evaluate_accuracy_one_hot(y_pred, y_test):

    def map_max_indices(nparray):

        def maxi(a):
            return a.argmax()
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    return {'accuracy': accuracy}
