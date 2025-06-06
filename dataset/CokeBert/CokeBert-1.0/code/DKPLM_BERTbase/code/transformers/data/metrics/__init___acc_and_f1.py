def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {'acc': acc, 'f1': f1, 'acc_and_f1': (acc + f1) / 2}
