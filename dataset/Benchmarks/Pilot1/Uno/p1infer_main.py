def main():
    parser = get_parser()
    args = parser.parse_args()
    prefix = args.prefix or os.path.basename(args.data)
    prefix = os.path.join(args.out_dir, prefix)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    df = pd.read_table(args.data, engine='c')
    df_x = df.copy()
    cat_cols = df.select_dtypes(['object']).columns
    if args.ignore_categoricals:
        df_x[cat_cols] = 0
    else:
        df_x[cat_cols] = df_x[cat_cols].apply(lambda x: x.astype('category'
            ).cat.codes)
    keepcols = args.keepcols
    ycol = args.ycol
    if ycol:
        if ycol.isdigit():
            ycol = df_x.columns[int(ycol)]
        df_x = df_x.drop(ycol, axis=1)
        keepcols = [ycol] + keepcols
    else:
        df_x = df_x
    if 'all' in keepcols:
        keepcols = list(df.columns)
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)
    x = df_x.as_matrix()
    y = model.predict(x)
    df_pred = df[keepcols]
    df_pred.insert(0, 'Pred', y)
    fname = '{}.predicted.tsv'.format(prefix)
    df_pred.to_csv(fname, sep='\t', index=False, float_format='%.3g')
    print('Predictions saved in {}\n'.format(fname))
