def adjust_alpha(gParameters, X_test, truths_test, labels_val, model, alpha,
    add_index):
    task_names = gParameters['task_names']
    task_list = gParameters['task_list']
    avg_loss = 0.0
    ret = []
    ret_k = []
    max_abs = gParameters['max_abs']
    min_acc = gParameters['min_acc']
    alpha_scale_factor = gParameters['alpha_scale_factor']
    feature_test = X_test
    loss = model.evaluate(feature_test, labels_val)
    avg_loss = avg_loss + loss[0]
    pred = model.predict(feature_test)
    abs_gain = gParameters['abs_gain']
    acc_gain = gParameters['acc_gain']
    accs = []
    abst = []
    for k in range(alpha.shape[0]):
        if k in task_list:
            truth_test = truths_test[:, k]
            alpha_k = K.eval(alpha[k])
            pred_classes = pred[k].argmax(axis=-1)
            true_classes = truth_test
            true = K.eval(K.sum(K.cast(K.equal(pred_classes, true_classes),
                'int64')))
            false = K.eval(K.sum(K.cast(K.not_equal(pred_classes,
                true_classes), 'int64')))
            abstain = K.eval(K.sum(K.cast(K.equal(pred_classes, add_index[k
                ] - 1), 'int64')))
            print(true, false, abstain)
            total = false + true
            tot_pred = total - abstain
            abs_acc = 0.0
            abs_frac = abstain / total
            if tot_pred > 0:
                abs_acc = true / tot_pred
            scale_k = alpha_scale_factor[k]
            min_scale = scale_k
            max_scale = 1.0 / scale_k
            acc_error = abs_acc - min_acc[k]
            acc_error = min(acc_error, 0.0)
            abs_error = abs_frac - max_abs[k]
            abs_error = max(abs_error, 0.0)
            new_scale = 1.0 + acc_gain * acc_error + abs_gain * abs_error
            new_scale = min(new_scale, max_scale)
            new_scale = max(new_scale, min_scale)
            print('Scaling factor: ', new_scale)
            K.set_value(alpha[k], new_scale * alpha_k)
            print_abs_stats(task_names[k], new_scale * alpha_k, true, false,
                abstain, max_abs[k])
            ret_k.append(truth_test)
            ret_k.append(pred)
            ret.append(ret_k)
            accs.append(abs_acc)
            abst.append(abs_frac)
        else:
            accs.append(1.0)
            accs.append(0.0)
    write_abs_stats(gParameters['output_dir'] + 'abs_stats.csv', alpha,
        accs, abst)
    return ret, alpha
