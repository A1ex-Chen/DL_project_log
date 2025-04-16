def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help=
        'like facebook/bart-large-cnn,t5-base, etc.')
    parser.add_argument('input_path', type=str, help='like cnn_dm/test.source')
    parser.add_argument('save_path', type=str, help='where to save summaries')
    parser.add_argument('--reference_path', type=str, required=False, help=
        'like cnn_dm/test.target')
    parser.add_argument('--score_path', type=str, required=False, default=
        'metrics.json', help='where to save metrics')
    parser.add_argument('--device', type=str, required=False, default=
        DEFAULT_DEVICE, help='cuda, cuda:1, cpu etc.')
    parser.add_argument('--prefix', type=str, required=False, default=None,
        help='will be added to the begininng of src examples')
    parser.add_argument('--task', type=str, default='summarization', help=
        'used for task_specific_params + metrics')
    parser.add_argument('--bs', type=int, default=8, required=False, help=
        'batch size')
    parser.add_argument('--n_obs', type=int, default=-1, required=False,
        help='How many observations. Defaults to all.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dump-args', action='store_true', help=
        'print the custom hparams with the results')
    parser.add_argument('--info', nargs='?', type=str, const=datetime_now(),
        help=
        "use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used."
        )
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f'parsed the following generate kwargs: {parsed_args}')
    examples = [(' ' + x.rstrip() if 't5' in args.model_name else x.rstrip(
        )) for x in open(args.input_path).readlines()]
    if args.n_obs > 0:
        examples = examples[:args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(
            f'score_path {args.score_path} will be overwritten unless you type ctrl-c.'
            )
    runtime_metrics = generate_summaries_or_translations(examples, args.
        save_path, args.model_name, batch_size=args.bs, device=args.device,
        fp16=args.fp16, task=args.task, prefix=args.prefix, **parsed_args)
    if args.reference_path is None:
        return {}
    score_fn = (calculate_bleu if 'translation' in args.task else
        calculate_rouge)
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()
        ][:len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    scores.update(runtime_metrics)
    if args.dump_args:
        scores.update(parsed_args)
    if args.info:
        scores['info'] = args.info
    if verbose:
        print(scores)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, 'w'))
    return scores
