def save_dataframe(metrics, filename, args):
    """Save F1 metrics"""
    df = pd.DataFrame(metrics, index=[0])
    path = Path(args.savepath).joinpath(f'f1/{filename}.csv')
    df.to_csv(path, index=False)
