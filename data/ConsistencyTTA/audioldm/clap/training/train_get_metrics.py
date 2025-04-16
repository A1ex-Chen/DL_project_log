def get_metrics(audio_features, text_features, logit_scale_a,
    audio_features_mlp=None, text_features_mlp=None, logit_scale_t=None,
    mlp_loss=False):
    metrics = {}
    if mlp_loss:
        a_logits_per_audio = (logit_scale_a * audio_features @
            text_features_mlp.t()).detach().cpu()
        a_logits_per_text = a_logits_per_audio.t().detach().cpu()
        t_logits_per_audio = (logit_scale_t * audio_features_mlp @
            text_features.t()).detach().cpu()
        t_logits_per_text = t_logits_per_audio.t().detach().cpu()
        labels = torch.arange(audio_features.shape[0]).long()
        total_loss = (F.cross_entropy(a_logits_per_audio, labels) + F.
            cross_entropy(a_logits_per_text, labels) + F.cross_entropy(
            t_logits_per_audio, labels) + F.cross_entropy(t_logits_per_text,
            labels)) / 4
        metrics[f'cumulative_loss'] = total_loss.item()
        metrics[f'num_samples'] = audio_features.shape[0]
        logits = {'audio_to_text': (a_logits_per_audio + t_logits_per_audio
            ) / 2, 'text_to_audio': (a_logits_per_text + t_logits_per_text) / 2
            }
        ground_truth = torch.arange(len(text_features)).view(-1, 1)
    else:
        logits_per_audio = (logit_scale_a * audio_features @ text_features.t()
            ).detach().cpu()
        logits_per_text = logits_per_audio.t().detach().cpu()
        labels = torch.arange(audio_features.shape[0]).long()
        total_loss = (F.cross_entropy(logits_per_audio, labels) + F.
            cross_entropy(logits_per_text, labels)) / 2
        metrics[f'cumulative_loss'] = total_loss.item()
        metrics[f'num_samples'] = audio_features.shape[0]
        logits = {'audio_to_text': logits_per_audio, 'text_to_audio':
            logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f'{name}_mean_rank'] = preds.mean() + 1
        metrics[f'{name}_median_rank'] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f'{name}_R@{k}'] = np.mean(preds < k)
        metrics[f'{name}_mAP@10'] = np.mean(np.where(preds < 10, 1 / (preds +
            1), 0.0))
    return metrics
