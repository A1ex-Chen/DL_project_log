def run(args):
    """ """
    data = load_dataset('PiC/' + args.dataset, args.data_subset)['test']
    system = SemanticSearch()
    model_config = os.path.join(ROOT_DIR, 'model_config.json')
    with open(model_config, 'r') as f:
        config = json.load(f)
    model_fpath = [x for x in config if x['scorer'] == args.scorer and x[
        'scorer_type'] == args.scorer_type][-1]['model_fpath']
    if model_fpath != '':
        model_fpath = os.path.join(ROOT_DIR, model_fpath)
    system.set_scorer(args.scorer, model_fpath, args.scorer_type)
    system.set_extractor(args.extractor, int(args.ngram_min), int(args.
        ngram_max))
    eval_results, run_results = run_eval(args, system, data)
    eval_results['dataset'] = args.dataset
    eval_results['data_subset'] = args.data_subset
    eval_results['scorer'] = args.scorer
    eval_results['scorer_type'] = args.scorer_type
    eval_results['extractor'] = args.extractor
    eval_results['oracle'] = args.oracle_candidates
    eval_results['contextual'] = args.contextual
    with open(os.path.join(args.outdir, 'configs.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    export_results(eval_results, run_results, args.outdir, args.contextual)
