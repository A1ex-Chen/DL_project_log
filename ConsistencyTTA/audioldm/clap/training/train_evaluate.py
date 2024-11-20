def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            return metrics
    device = torch.device(args.device)
    model.eval()
    if is_master(args):
        print('Evaluating...')
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if args.val_dataset_names == ['Clotho', 'audiocaps']:
        if args.parallel_eval:
            raise NotImplementedError(
                'Parallel evaluation not supported for eval only Clotho and audiocaps.'
                )
        val_metrics_per_dataset = evaluate_clotho_audiocaps(model, data,
            epoch, args, autocast, device, tb_writer)
        for m in val_metrics_per_dataset.values():
            metrics.update(m)
        if 'epoch' not in metrics.keys():
            metrics.update({'epoch': epoch})
        metrics = select_top_metric_clotho_audiocaps(metrics,
            val_metrics_per_dataset, args)
    elif 'val' in data and (args.val_frequency and (epoch % args.
        val_frequency == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples
        eval_info = {}
        if args.clap_mlploss:
            eval_info['all'] = {'cumulative_loss': 0.0, 'num_samples': 0,
                'all_audio_features': [], 'all_text_features': [],
                'all_audio_features_mlp': [], 'all_text_features_mlp': []}
        else:
            eval_info['all'] = {'cumulative_loss': 0.0, 'num_samples': 0,
                'all_audio_features': [], 'all_text_features': []}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audios = batch
                texts = batch['text']
                all_names = list(set(['-'.join(b.split('/')[-3:-1]) for b in
                    batch['__url__']]))
                for name in all_names:
                    if name not in eval_info.keys():
                        if args.clap_mlploss:
                            eval_info[name] = {'cumulative_loss': 0.0,
                                'num_samples': 0, 'all_audio_features': [],
                                'all_text_features': [],
                                'all_audio_features_mlp': [],
                                'all_text_features_mlp': []}
                        else:
                            eval_info[name] = {'cumulative_loss': 0.0,
                                'num_samples': 0, 'all_audio_features': [],
                                'all_text_features': []}
                with autocast():
                    (audio_features, text_features, audio_features_mlp,
                        text_features_mlp, logit_scale_a, logit_scale_t
                        ) = model(audios, texts, device)
                    if args.parallel_eval:
                        if args.clap_mlploss:
                            (audio_features, text_features,
                                audio_features_mlp, text_features_mlp) = (
                                gather_features(audio_features=
                                audio_features, text_features=text_features,
                                audio_features_mlp=audio_features_mlp,
                                text_features_mlp=text_features_mlp,
                                local_loss=False, gather_with_grad=False,
                                rank=args.rank, world_size=args.world_size,
                                use_horovod=args.horovod, mlp_loss=args.
                                clap_mlploss))
                        else:
                            audio_features, text_features = gather_features(
                                audio_features=audio_features,
                                text_features=text_features, local_loss=
                                False, gather_with_grad=False, rank=args.
                                rank, world_size=args.world_size,
                                use_horovod=args.horovod, mlp_loss=args.
                                clap_mlploss)
                    if is_master(args):
                        num_samples += audio_features.shape[0]
                        for n in [*all_names, 'all']:
                            if n == 'all':
                                eval_info[n]['all_audio_features'].append(
                                    audio_features.cpu())
                                eval_info[n]['all_text_features'].append(
                                    text_features.cpu())
                                if args.clap_mlploss:
                                    eval_info[n]['all_audio_features_mlp'
                                        ].append(audio_features_mlp.cpu())
                                    eval_info[n]['all_text_features_mlp'
                                        ].append(text_features_mlp.cpu())
                            else:
                                idx = np.where(np.array(['-'.join(b.split(
                                    '/')[-3:-1]) for b in batch['__url__']]
                                    ) == n)[0]
                                eval_info[n]['all_audio_features'].append(
                                    audio_features.cpu().index_select(0,
                                    torch.tensor(idx).long()))
                                eval_info[n]['all_text_features'].append(
                                    text_features.cpu().index_select(0,
                                    torch.tensor(idx).long()))
                                if args.clap_mlploss:
                                    eval_info[n]['all_audio_features_mlp'
                                        ].append(audio_features_mlp.cpu().
                                        index_select(0, torch.tensor(idx).
                                        long()))
                                    eval_info[n]['all_text_features_mlp'
                                        ].append(text_features_mlp.cpu().
                                        index_select(0, torch.tensor(idx).
                                        long()))
                if is_master(args) and i % 100 == 0:
                    logging.info(
                        f'Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]'
                        )
            if is_master(args):
                val_metrics_per_dataset = {}
                for n in eval_info.keys():
                    if args.clap_mlploss:
                        metrics_single_dataset = get_metrics(audio_features
                            =torch.cat(eval_info[n]['all_audio_features']),
                            text_features=torch.cat(eval_info[n][
                            'all_text_features']), logit_scale_a=
                            logit_scale_a.cpu(), audio_features_mlp=torch.
                            cat(eval_info[n]['all_audio_features_mlp']),
                            text_features_mlp=torch.cat(eval_info[n][
                            'all_text_features_mlp']), logit_scale_t=
                            logit_scale_t.cpu(), mlp_loss=args.clap_mlploss)
                    else:
                        metrics_single_dataset = get_metrics(audio_features
                            =torch.cat(eval_info[n]['all_audio_features']),
                            text_features=torch.cat(eval_info[n][
                            'all_text_features']), logit_scale_a=
                            logit_scale_a.cpu(), mlp_loss=args.clap_mlploss)
                    val_metrics_per_dataset[n] = {(n + '/' + k): v for k, v in
                        metrics_single_dataset.items()}
                    metrics.update(val_metrics_per_dataset[n])
                    if 'epoch' not in metrics.keys():
                        metrics.update({'epoch': epoch})
    if is_master(args):
        if not metrics:
            return metrics
        logging.info(f'Eval Epoch: {epoch} ' + '\n'.join(['\t'.join([
            f'{k}: {round(v, 4):.4f}' for k, v in m.items()]) for m in
            val_metrics_per_dataset.values()]))
        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f'val/{name}', val, epoch)
            with open(os.path.join(args.checkpoint_path, 'results.jsonl'), 'a+'
                ) as f:
                f.write(json.dumps(metrics))
                f.write('\n')
        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            for name, val in metrics.items():
                wandb.log({f'val/{name}': val, 'epoch': epoch})
        return metrics
    else:
        return metrics
