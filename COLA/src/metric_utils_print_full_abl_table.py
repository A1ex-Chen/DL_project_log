def print_full_abl_table(df, configs):
    for dataset in df.dataset_name.unique():
        for score in ['score_dl1', 'score_dl2']:
            dt = df.query(f"dataset_name=='{dataset}' and score == '{score}'")
            res = []
            for d, f, s, q, c, e in configs:
                dtt = ablation_query(dt, d, f, s, q, c, e)
                res.append(dtt.acc.max())
            res = np.array(res).astype(float)
            res_min, res_max = np.min(res), np.max(res)
            min_idx, max_idx = np.where(res == res_min)[0], np.where(res ==
                res_max)[0]
            res_str = [f'${r:.3f}$' for r in res]
            best = res_str[max_idx[0]]
            worst = res_str[min_idx[0]]
            for idx in min_idx:
                res_str[idx] = '\\abpoor{' + res_str[idx] + '}'
            for idx in max_idx:
                res_str[idx] = '\\abgood{' + res_str[idx] + '}'
            res_str = [best, worst] + res_str
            print(f"{dataset}/{score}: {'&'.join(f' {r} ' for r in res_str)}\n"
                )
