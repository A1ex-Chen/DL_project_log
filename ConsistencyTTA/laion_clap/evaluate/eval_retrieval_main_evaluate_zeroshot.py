def evaluate_zeroshot(model, data, start_epoch, args, writer):
    dataloader = data['val'].dataloader
    metrics = {}
    device = torch.device(args.device)
    model.eval()
    metrics.update({'epoch': start_epoch})
    all_audio_features = []
    all_class_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            audios = batch
            audio_features = model(audios, None, device)
            audio_features = F.normalize(audio_features, dim=-1)
            all_audio_features.append(audio_features.detach().cpu())
            all_class_labels.append(torch.argmax(batch['class_label'], 1).
                long())
        all_audio_features = torch.cat(all_audio_features, dim=0)
        all_class_labels = torch.cat(all_class_labels, dim=0)
        metrics['num_samples'] = all_audio_features.shape[0]
        all_texts = [('This is a sound of ' + t) for t in args.
            class_index_dict.keys()]
        if args.tmodel == 'transformer':
            from clap_module.tokenizer import tokenize
            all_texts = tokenize(all_texts)
        else:
            from training.data import tokenizer
            all_texts = tokenizer(all_texts)
        all_text_features = model(None, all_texts, device)
        all_text_features = F.normalize(all_text_features, dim=-1).detach(
            ).cpu()
        logit_scale_a, logit_scale_t = model(None, None, device)
        logit_scale_a = logit_scale_a.cpu()
        logits_per_audio = (logit_scale_a * all_audio_features @
            all_text_features.t()).detach().cpu()
        logits_per_text = logits_per_audio.t().detach().cpu()
        ground_truth = all_class_labels.view(-1, 1)
        logit = logits_per_audio
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f'{args.datasetnames[0]}_mean_rank'] = preds.mean() + 1
        metrics[f'{args.datasetnames[0]}_median_rank'] = np.floor(np.median
            (preds)) + 1
        for k in [1, 5, 10]:
            metrics[f'{args.datasetnames[0]}_R@{k}'] = np.mean(preds < k)
        metrics[f'{args.datasetnames[0]}_mAP@10'] = np.mean(np.where(preds <
            10, 1 / (preds + 1), 0.0))
        logging.info(f'Eval Epoch: {start_epoch} ' + '\t'.join([
            f'{k}: {round(v, 4):.4f}' for k, v in metrics.items()]))
        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            for name, val in metrics.items():
                wandb.log({f'val/{name}': val, 'epoch': start_epoch})
