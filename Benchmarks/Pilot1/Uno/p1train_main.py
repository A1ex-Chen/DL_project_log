def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    prefix = args.prefix or os.path.basename(args.data)
    prefix = os.path.join(args.out_dir, prefix)
    df = pd.read_table(args.data, engine='c', sep=',' if args.csv else '\t')
    x, y, splits, features = split_data(df, ycol=args.ycol, classify=args.
        classify, cv=args.cv, bins=args.bins, cutoffs=args.cutoffs,
        groupcols=args.groupcols, ignore_categoricals=args.
        ignore_categoricals, verbose=True)
    if args.classify and len(np.unique(y)) < 2:
        print('Not enough classes\n')
        return
    best_score, best_model = -np.Inf, None
    for model in args.models:
        if args.classify:
            class_weight = 'balanced' if args.balanced else None
            score = classify(model, x, y, splits, features, threads=args.
                threads, prefix=prefix, seed=args.seed, class_weight=
                class_weight)
        else:
            score = regress(model, x, y, splits, features, threads=args.
                threads, prefix=prefix, seed=args.seed)
        if score >= best_score:
            best_score = score
            best_model = model
    print('Training the best model ({}={:.3g}) on the entire dataset...'.
        format(best_model, best_score))
    name = 'best.classifier' if args.classify else 'best.regressor'
    fname = train(best_model, x, y, features, classify=args.classify,
        threads=args.threads, prefix=prefix, name=name, save=True)
    print('Model saved in {}\n'.format(fname))
