def update_top_k_performance(new_metrics_inputs, current_top_k_ckpt_metrics,
    args, ckpt, bignumbetter=True, pretrain_epoch=0):
    """
    Record the top-k performance of the current epoch.
    current_top_k_metrics is a dictionary of the form: {1: top_1_ckpt_measure, 2: top_2_ckpt_measure, ...}
    """
    if isinstance(new_metrics_inputs, (list, tuple)):
        new_metrics_inputs = np.mean(new_metrics_inputs)
        return update_top_k_performance(new_metrics_inputs,
            current_top_k_ckpt_metrics, args=args, ckpt=ckpt, bignumbetter=
            bignumbetter, pretrain_epoch=pretrain_epoch)
    elif isinstance(new_metrics_inputs, dict):
        new_metrics_inputs = np.mean(list(new_metrics_inputs.values()))
        return update_top_k_performance(new_metrics_inputs,
            current_top_k_ckpt_metrics, args=args, ckpt=ckpt, bignumbetter=
            bignumbetter, pretrain_epoch=pretrain_epoch)
    elif isinstance(new_metrics_inputs, (float, int)):
        update_flag = {k: (False) for k in current_top_k_ckpt_metrics.keys()}
        sorted_keys = sorted(current_top_k_ckpt_metrics.keys())
        sorted_values = sorted(current_top_k_ckpt_metrics.values(), reverse
            =bignumbetter)
        sorted_values_ = copy.deepcopy(sorted_values)
        sorted_values.append(new_metrics_inputs)
        sorted_values = sorted(sorted_values, reverse=bignumbetter)
        sorted_values = sorted_values[:-1]
        if sorted_values == sorted_values_:
            return current_top_k_ckpt_metrics, new_metrics_inputs
        else:
            for i in range(len(sorted_keys)):
                if current_top_k_ckpt_metrics[sorted_keys[i]] != sorted_values[
                    i]:
                    current_top_k_ckpt_metrics[sorted_keys[i]] = sorted_values[
                        i]
                    update_flag[sorted_keys[i]] = True
            for i in range(len(update_flag)):
                if update_flag[i]:
                    maintain_ckpts(args, i, len(sorted_keys))
                    torch.save(ckpt, os.path.join(args.checkpoint_path,
                        f'pretrain_epoch_{pretrain_epoch}_lp_epoch_top_{i}.pt')
                        )
                    break
            return current_top_k_ckpt_metrics, new_metrics_inputs
