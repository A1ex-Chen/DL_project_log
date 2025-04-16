def multitask_accuracy(target, output):
    """Compute the accuracy for multitask problems"""
    accuracies = {}
    for key, value in target.items():
        accuracies[key] = accuracy(target[key], output[key])
    return accuracies
