def evaluate(model, data, epoch, args, tb_writer=None, extra_suffix=''):
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            return metrics
    device = torch.device(args.device)
    model.eval()
    if is_master(args):
        print('Evaluating...')
        metric_names = args.lp_metrics.split(',')
        eval_tool = LPMetrics(metric_names=metric_names)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and (epoch % args.
        val_frequency == 0 or epoch == args.epochs)):
        if args.parallel_eval:
            dataloader, sampler = data['val'].dataloader, data['val'].sampler
            if args.distributed and sampler is not None:
                sampler.set_epoch(epoch)
            samples_per_val = dataloader.num_samples
        else:
            dataloader = data['val'].dataloader
            num_samples = 0
            samples_per_val = dataloader.num_samples
        eval_info = {'pred': [], 'target': []}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audio = batch
                class_label = batch['class_label']
                class_label = class_label.to(device=device, non_blocking=True)
                with autocast():
                    pred = model(audio, device=device)
                    if args.parallel_eval:
                        pred, class_label = lp_gather_features(pred,
                            class_label, args.world_size, args.horovod)
                    eval_info['pred'].append(pred)
                    eval_info['target'].append(class_label)
                num_samples += class_label.shape[0]
                if i % 100 == 0:
                    logging.info(
                        f'Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]'
                        )
            if is_master(args):
                eval_info['pred'] = torch.cat(eval_info['pred'], 0).cpu()
                eval_info['target'] = torch.cat(eval_info['target'], 0).cpu()
                metric_dict = eval_tool.evaluate_mertics(eval_info['pred'],
                    eval_info['target'])
                metrics.update(metric_dict)
                if 'epoch' not in metrics.keys():
                    metrics.update({'epoch': epoch})
    if is_master(args):
        if not metrics:
            return metrics
        logging.info(f'Eval Epoch: {epoch} ' + '\n'.join(['\t'.join([
            f'{m}: {round(metrics[m], 4):.4f}']) for m in metrics]))
        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f'val{extra_suffix}/{name}', val,
                        epoch)
            with open(os.path.join(args.checkpoint_path, 'results.jsonl'), 'a+'
                ) as f:
                f.write(json.dumps(metrics))
                f.write('\n')
        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            for name, val in metrics.items():
                wandb.log({f'val{extra_suffix}/{name}': val, 'epoch': epoch})
        return metrics
    else:
        return metrics
