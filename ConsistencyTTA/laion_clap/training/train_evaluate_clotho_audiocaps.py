def evaluate_clotho_audiocaps(model, data, epoch, args, autocast, device,
    tb_writer=None):
    """
    Adapted from https://github.com/XinhaoMei/audio-text_retrieval/blob/main/tools/utils.py.
    1. for text-to-audio retrieval, do 5 times and average the results
    2. for R@1, R@5, R@10 in audio-to-text retrieval, take the best rank among 5 text
    3. for map@10 in audio-to-text retrieval:
        3.1: sort the rank of 5 text
        3.2: exclude the rank >=10 (0-index)
        3.3: compute the map regarding the remaining ranks: np.mean(np.arange(1, len(ranks)+1) / ranks).
        (3.3) That is, take the top ranks of 5 text that is < 10, and assign the descending number as ground truth.
        (3.3) E.g.: the ground truth of first rank of the 5 text should be 1, the second rank should be 2, etc.
    """
    dataloader = data['val'].dataloader
    with torch.no_grad():
        eval_info = {}
        for i, batch in enumerate(dataloader):
            audios = batch
            if args.tmodel == 'transformer':
                from clap_module import tokenize
                texts = [tokenize(t) for t in batch['full_text']]
                texts = torch.cat(texts)
            else:
                from .data import tokenizer
                texts = [tokenizer(t, tmodel=args.tmodel) for t in batch[
                    'full_text']]
                texts = {k: torch.cat([t[k] for t in texts]) for k in texts
                    [0].keys()}
            all_names = list(set(['-'.join(b.split('/')[-3:-1]) for b in
                batch['__url__']]))
            for name in all_names:
                if name not in eval_info.keys():
                    eval_info[name] = {'cumulative_loss': 0.0,
                        'num_samples': 0, 'all_audio_features': [],
                        'all_text_features': []}
            with autocast():
                audio_features = model(audios, None, device)
                text_features = model(None, texts, device)
                audio_features = F.normalize(audio_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                all_names = list(set(['-'.join(b.split('/')[-3:-1]) for b in
                    batch['__url__']]))
                for n in all_names:
                    idx = np.where(np.array(['-'.join(b.split('/')[-3:-1]) for
                        b in batch['__url__']]) == n)[0]
                    eval_info[n]['all_audio_features'].append(audio_features
                        .cpu().index_select(0, torch.tensor(idx).long()))
                    eval_info[n]['all_text_features'].append(text_features.
                        cpu().reshape([-1, 5, text_features.shape[1]]).
                        index_select(0, torch.tensor(idx).long()).reshape([
                        -1, text_features.shape[1]]))
        val_metrics_all = {}
        for n in eval_info.keys():
            logit_scale_a, logit_scale_t = model(None, None, device)
            logit_scale_a = logit_scale_a.cpu()
            audio_features = torch.cat(eval_info[n]['all_audio_features'],
                dim=0)
            text_features = torch.cat(eval_info[n]['all_text_features'], dim=0)
            logits_per_audio = (logit_scale_a * audio_features @
                text_features.t()).detach().cpu()
            logits_per_text = logits_per_audio.t().detach().cpu()
            logging.info(
                f'dataset {n}, logits_per_audio shape: {logits_per_audio.shape}, logits_per_text shape: {logits_per_text.shape}'
                )
            metrics = {}
            num_samples = audio_features.shape[0]
            metrics[f'num_samples'] = num_samples
            labels = torch.arange(audio_features.shape[0]).long()
            audio_to_text_loss = [F.cross_entropy(logits_per_audio.reshape(
                num_samples, num_samples, 5)[:, :, d], labels) for d in
                range(5)]
            text_to_audio_loss = [F.cross_entropy(logits_per_text.reshape(
                num_samples, 5, num_samples)[:, d, :], labels) for d in
                range(5)]
            total_loss = (np.mean(audio_to_text_loss) + np.mean(
                text_to_audio_loss)) / 2
            metrics[f'cumulative_loss'] = total_loss.item()
            pred_text = []
            for d in range(5):
                logit = logits_per_text.reshape(num_samples, 5, num_samples)[
                    :, d, :]
                ground_truth = torch.arange(len(logit)).view(-1, 1)
                ranking = torch.argsort(logit, descending=True)
                preds = torch.where(ranking == ground_truth)[1]
                pred_text.append(preds.detach().cpu().numpy())
            pred_text_concat = np.concatenate(pred_text, axis=0)
            metrics[f'text_to_audio_mean_rank'] = pred_text_concat.mean() + 1
            metrics[f'text_to_audio_median_rank'] = np.floor(np.median(
                pred_text_concat)) + 1
            for k in [1, 5, 10]:
                metrics[f'text_to_audio_R@{k}'] = np.mean(pred_text_concat < k)
            metrics[f'text_to_audio_mAP@10'] = np.mean(np.where(
                pred_text_concat < 10, 1 / (pred_text_concat + 1), 0.0))
            map_all = []
            pred_audio_all = []
            for d in range(num_samples):
                logit_single = logits_per_audio[d, :]
                ranking = torch.argsort(logit_single, descending=True)
                ground_truth = torch.arange(d * 5, d * 5 + 5)[None]
                all_pred = torch.where(torch.stack([ranking] * 5) ==
                    ground_truth.view(-1, 1))[1]
                min_pred = torch.min(all_pred)
                pred_audio_all.append(min_pred.detach().cpu().numpy())
                all_pred_filter = all_pred[all_pred < 10].detach().cpu().numpy(
                    )
                map_single = np.sum(np.arange(1, len(all_pred_filter) + 1) /
                    (all_pred_filter + 1)) / 5
                map_all.append(map_single)
            metrics[f'audio_to_text_mAP@10'] = np.mean(map_all)
            for k in [1, 5, 10]:
                metrics[f'audio_to_text_R@{k}'] = np.mean(np.array(
                    pred_audio_all) < k)
            val_metrics_all[n] = {(n + '/' + k): v for k, v in metrics.items()}
    return val_metrics_all
