def run_generate():
    parser = argparse.ArgumentParser(epilog=
        'Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate'
        )
    parser.add_argument('--data_dir', type=str, help='like cnn_dm/test.source')
    parser.add_argument('--model_name', type=str, help=
        'like facebook/bart-large-cnn,t5-base, etc.', default=
        'sshleifer/distilbart-xsum-12-3')
    parser.add_argument('--save_dir', type=str, help='where to save',
        default='tmp_gen')
    parser.add_argument('--max_source_length', type=int, default=None)
    parser.add_argument('--type_path', type=str, default='test', help=
        'which subset to evaluate typically train/val/test')
    parser.add_argument('--task', type=str, default='summarization', help=
        'used for task_specific_params + metrics')
    parser.add_argument('--bs', type=int, default=8, required=False, help=
        'batch size')
    parser.add_argument('--local_rank', type=int, default=-1, required=
        False, help='should be passed by distributed.launch')
    parser.add_argument('--n_obs', type=int, default=None, required=False,
        help='How many observations. Defaults to all.')
    parser.add_argument('--num_return_sequences', type=int, default=1,
        required=False, help='How many sequences to return')
    parser.add_argument('--sync_timeout', type=int, default=600, required=
        False, help=
        'How long should master process wait for other processes to finish.')
    parser.add_argument('--src_lang', type=str, default=None, required=False)
    parser.add_argument('--tgt_lang', type=str, default=None, required=False)
    parser.add_argument('--prefix', type=str, required=False, default=None,
        help='will be added to the begininng of src examples')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--debug', action='store_true')
    start_time = time.time()
    args, rest = parser.parse_known_args()
    generate_kwargs = parse_numeric_n_bool_cl_kwargs(rest)
    if generate_kwargs and args.local_rank <= 0:
        print(f'parsed the following generate kwargs: {generate_kwargs}')
    json_save_dir = Path(args.save_dir + '_tmp')
    Path(json_save_dir).mkdir(exist_ok=True)
    intermediate_files = list(json_save_dir.glob('rank_*.json'))
    if intermediate_files:
        raise ValueError(
            f'Found files at {json_save_dir} please move or remove them.')
    dataset_kwargs = {}
    if args.src_lang is not None:
        dataset_kwargs['src_lang'] = args.src_lang
    if args.tgt_lang is not None:
        dataset_kwargs['tgt_lang'] = args.tgt_lang
    Path(args.save_dir).mkdir(exist_ok=True)
    results, num_replicas = eval_data_dir(args.data_dir, json_save_dir,
        args.model_name, type_path=args.type_path, bs=args.bs, fp16=args.
        fp16, task=args.task, local_rank=args.local_rank, n_obs=args.n_obs,
        max_source_length=args.max_source_length, num_return_sequences=args
        .num_return_sequences, prefix=args.prefix, dataset_kwargs=
        dataset_kwargs, **generate_kwargs)
    if args.local_rank <= 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)
        partial_results = gather_results_from_each_node(num_replicas,
            json_save_dir, args.sync_timeout)
        preds = combine_partial_results(partial_results)
        if args.num_return_sequences > 1:
            save_path = save_dir.joinpath('pseudolabel_results.json')
            print(
                f'Saving aggregated results at {save_path}, intermediate in {json_save_dir}/'
                )
            save_json(preds, save_path)
            return
        tgt_file = Path(args.data_dir).joinpath(args.type_path + '.target')
        labels = [x.rstrip() for x in open(tgt_file).readlines()][:len(preds)]
        calc_bleu = 'translation' in args.task
        score_fn = calculate_bleu if calc_bleu else calculate_rouge
        metric_name = 'bleu' if calc_bleu else 'rouge'
        metrics: Dict = score_fn(preds, labels)
        metrics['n_obs'] = len(preds)
        runtime = time.time() - start_time
        metrics['seconds_per_sample'] = round(runtime / metrics['n_obs'], 4)
        metrics['n_gpus'] = num_replicas
        metrics_save_path = save_dir.joinpath(
            f'{args.type_path}_{metric_name}.json')
        save_json(metrics, metrics_save_path, indent=None)
        print(metrics)
        write_txt_file(preds, save_dir.joinpath(
            f'{args.type_path}_generations.txt'))
        if args.debug:
            write_txt_file(labels, save_dir.joinpath(
                f'{args.type_path}.target'))
        else:
            shutil.rmtree(json_save_dir)
