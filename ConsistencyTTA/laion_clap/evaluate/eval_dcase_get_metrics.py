def get_metrics(text_to_audio_logits):
    metrics = {}
    ground_truth = torch.repeat_interleave(torch.arange(len(text_features) //
        5), 5).view(-1, 1)
    ranking = torch.argsort(text_to_audio_logits, descending=True)
    preds = torch.where(ranking == ground_truth)[1]
    preds = preds.detach().cpu().numpy()
    metrics[f'mean_rank'] = preds.mean() + 1
    metrics[f'median_rank'] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f'R@{k}'] = np.mean(preds < k)
    metrics[f'mAP@10'] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))
    return metrics
