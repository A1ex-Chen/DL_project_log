def ram_report(df, topk='top1', verbose=False, export=False, filename='report'
    ):
    if verbose:
        idx_max = df.index[df.ram == df.ram.max()]
        print('-' * 120)
        print(df.to_string())
        print('-' * 120)
        print(
            f' >> Peak Memory of {df.ram.max() / 2 ** 10:.0f} kB found in the following node(s):'
            )
        if topk == 'top1':
            for idx in idx_max:
                print(df.loc[idx].to_string())
                print()
    if export:
        export_path = f'{filename}.csv'
        df.to_csv(export_path)
        print(f'RAM usage report exported to {export_path}')
