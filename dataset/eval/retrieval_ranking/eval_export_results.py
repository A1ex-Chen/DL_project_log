def export_results(eval_results, run_results, outdir, contextual):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, 'eval_results.json'), 'w') as fp:
        json.dump(eval_results, fp, indent=4)
    with open(os.path.join(outdir, 'run_results.json'), 'w') as fp:
        json.dump(run_results, fp, indent=4)
    subdir = 'contextual' if contextual else 'non_contextual'
    summary_dir = outdir.split(subdir)[0]
    with open(os.path.join(summary_dir, subdir, 'summary.txt'), 'a') as f:
        f.write(outdir + '\t')
        f.write(str(eval_results['Top@1']) + '\t')
        f.write(str(eval_results['Top@3']) + '\t')
        f.write(str(eval_results['Top@5']) + '\t')
        f.write(str(eval_results['MRR@5']) + '\t')
