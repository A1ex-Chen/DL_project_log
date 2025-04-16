def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=
        'Run models with given dataloader', allow_abbrev=False)
    parser.add_argument('--metrics', help=
        f'Path to python module containing metrics calculator', required=True)
    parser.add_argument('--csv', help='Path to csv file', required=True)
    parser.add_argument('--dump-dir', help=
        'Path to directory with dumped outputs (and labels)', required=True)
    args, *_ = parser.parse_known_args()
    MetricsCalculator = load_from_file(args.metrics, 'metrics',
        'MetricsCalculator')
    ArgParserGenerator(MetricsCalculator).update_argparser(parser)
    args = parser.parse_args()
    LOGGER.info(f'args:')
    for key, value in vars(args).items():
        LOGGER.info(f'    {key} = {value}')
    MetricsCalculator = load_from_file(args.metrics, 'metrics',
        'MetricsCalculator')
    metrics_calculator: BaseMetricsCalculator = ArgParserGenerator(
        MetricsCalculator).from_args(args)
    ids = get_data(args.dump_dir, 'ids')['ids']
    x = get_data(args.dump_dir, 'inputs')
    y_true = get_data(args.dump_dir, 'labels')
    y_pred = get_data(args.dump_dir, 'outputs')
    common_keys = list({k for k in y_true or []} & {k for k in y_pred or []})
    for key in common_keys:
        if y_true[key].shape != y_pred[key].shape:
            LOGGER.warning(
                f'Model predictions and labels shall have equal shapes. y_pred[{key}].shape={y_pred[key].shape} != y_true[{key}].shape={y_true[key].shape}'
                )
    metrics = metrics_calculator.calc(ids=ids, x=x, y_pred=y_pred, y_real=
        y_true)
    metrics = {TOTAL_COLUMN_NAME: len(ids), **metrics}
    metric_names_with_space = [name for name in metrics if any([(c in
        string.whitespace) for c in name])]
    if metric_names_with_space:
        raise ValueError(
            f"Metric names shall have no spaces; Incorrect names: {', '.join(metric_names_with_space)}"
            )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
