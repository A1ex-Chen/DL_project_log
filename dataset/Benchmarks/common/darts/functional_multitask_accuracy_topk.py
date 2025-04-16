def multitask_accuracy_topk(target, output, topk=(1,)):
    """Compute the topk accuracy for multitask problems"""
    topk_accuracies = {}
    for key, value in target.items():
        topk_accuracies[key] = accuracy_topk(output[key], target[key], topk)
    return topk_accuracies
