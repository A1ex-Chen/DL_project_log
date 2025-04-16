def eval_motChallenge(matrixes_predictions, paths, metrics=None,
    generate_overall=True):
    gt = OrderedDict([(os.path.split(p)[1], mm.io.loadtxt(os.path.join(p,
        'gt/gt.txt'), fmt='mot15-2D', min_confidence=1)) for p in paths])
    ts = OrderedDict([(os.path.split(paths[n])[1], load_motchallenge(
        matrixes_predictions[n])) for n in range(len(paths))])
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)
    if metrics is None:
        metrics = list(mm.metrics.motchallenge_metrics)
    mm.lap.default_solver = 'scipy'
    print('Computing metrics...')
    summary_dataframe = mh.compute_many(accs, names=names, metrics=metrics,
        generate_overall=generate_overall)
    summary_text = mm.io.render_summary(summary_dataframe, formatters=mh.
        formatters, namemap=mm.io.motchallenge_metric_names)
    return summary_text, summary_dataframe
