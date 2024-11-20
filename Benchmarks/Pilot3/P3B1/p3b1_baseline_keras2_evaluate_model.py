def evaluate_model(X_test, truths_test, labels_test, models):
    avg_loss = 0.0
    ret = []
    for k in range(len(models)):
        ret_k = []
        feature_test = X_test[k]
        truth_test = truths_test[k]
        label_test = labels_test[k]
        model = models[k]
        loss = model.evaluate(feature_test, label_test)
        avg_loss = avg_loss + loss[0]
        print('In EVALUATE loss: ', loss)
        pred = model.predict(feature_test)
        ret_k.append(truth_test)
        ret_k.append(np.argmax(pred, axis=1))
        ret.append(ret_k)
    avg_loss = avg_loss / float(len(models))
    ret.append(avg_loss)
    return ret
