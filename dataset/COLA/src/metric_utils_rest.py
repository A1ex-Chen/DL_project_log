import numpy as np
import ast
import pathos
import pandas as pd
import src.metrics as metrics









    # return np.mean(packed_data['label'] == dat.groupby(['index']).apply(lambda s: s.iloc[s[col_name].argmax()].label_idx))












### Ablation on N



### Analysis of full examples







### full ablations







    dat[col_name] = dat.apply(lambda s: func(*get_probs(s), eps=eps, **kwargs)
                              , axis=1)
    return np.mean(dat.groupby(['index']).apply(lambda s: s.answer_idx.iloc[s[col_name].argmax()]) == 1)


def get_metrics(epss, test_func, dat, **kwargs):
    accs_bar = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=f"score_dl1_{eps}",
                          **kwargs) for eps in epss]
    accs_l2 = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=f"score_dl2_{eps}",
                         ord=2, **kwargs) for eps in epss]

    accs_pdy = [test_func(metrics.delta_pdy, dat=dat, col_name='score_de1', **kwargs)] * len(epss)
    accs_palldy = [test_func(metrics.delta_palldy, dat=dat, col_name='score_da', **kwargs)] * len(epss)
    accs_pxy = [test_func(metrics.delta_pxd, dat=dat, col_name='score_dx', **kwargs)] * len(epss)
    return dat, epss, accs_bar, accs_l2, accs_pdy, accs_palldy, accs_pxy


def get_metrics_todf(epss, test_func, dat, **kwargs):
    accs_bar = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name='tmp_score',
                          **kwargs) for eps in epss]

    accs_l2 = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name='tmp_score',
                         ord=2, **kwargs) for eps in epss]

    df1 = pd.DataFrame(np.array([epss, accs_bar]).T, columns=['eps', 'acc'])
    df2 = pd.DataFrame(np.array([epss, accs_l2]).T, columns=['eps', 'acc'])
    df1['score'] = 'score_dl1'
    df2['score'] = 'score_dl2'
    df = pd.concat([df1, df2])

    accs_pdy = test_func(metrics.delta_pdy, dat=dat, col_name='tmp_score', **kwargs)
    accs_palldy = test_func(metrics.delta_palldy, dat=dat, col_name='tmp_score', **kwargs)
    accs_pxy = test_func(metrics.delta_pxd, dat=dat, col_name='tmp_score', **kwargs)
    df = pd.concat([df, pd.DataFrame([
        [-1, accs_pdy, 'score_de1'],
        [-1, accs_palldy, 'score_da'],
        [-1, accs_pxy, 'score_dx']
    ], columns=['eps', 'acc', 'score'])])
    return df


def get_metrics_todf_mult(epss, test_func, dat, n_workers=30, **kwargs):


    with pathos.multiprocessing.ProcessingPool(n_workers) as pool:
        accs_bar = pool.map(dl1_func, epss)
        accs_l2 = pool.map(dl2_func, epss)

    df1 = pd.DataFrame(np.array([epss, accs_bar]).T, columns=['eps', 'acc'])
    df2 = pd.DataFrame(np.array([epss, accs_l2]).T, columns=['eps', 'acc'])
    df1['score'] = 'score_dl1'
    df2['score'] = 'score_dl2'
    df = pd.concat([df1, df2])

    accs_pdy = test_func(metrics.delta_pdy, dat=dat, col_name='tmp_score', **kwargs)
    accs_palldy = test_func(metrics.delta_palldy, dat=dat, col_name='tmp_score', **kwargs)
    accs_pxy = test_func(metrics.delta_pxd, dat=dat, col_name='tmp_score', **kwargs)
    df = pd.concat([df, pd.DataFrame([
        [-1, accs_pdy, 'score_de1'],
        [-1, accs_palldy, 'score_da'],
        [-1, accs_pxy, 'score_dx']
    ], columns=['eps', 'acc', 'score'])])
    return df


### Ablation on N

def get_metrics_todf_N_mult(epss_input, test_func, dat, n_smp, n_rep=50, n_workers=30, **kwargs):






    epss = np.repeat(epss_input, n_rep)
    dummies = np.arange(n_smp)
    with pathos.multiprocessing.ProcessingPool(n_workers) as pool:
        accs_bar = pool.map(dl1_func, epss)
        accs_l2 = pool.map(dl2_func, epss)
        accs_pdy = pool.map(pdy_func, dummies)
        accs_palldy = pool.map(palldy_func, dummies)
        accs_pxy = pool.map(pxy_func, dummies)

    df1 = pd.DataFrame(np.array([epss, accs_bar]).T, columns=['eps', 'acc'])
    df2 = pd.DataFrame(np.array([epss, accs_l2]).T, columns=['eps', 'acc'])
    df1['score'] = 'score_dl1'
    df2['score'] = 'score_dl2'
    df = pd.concat([df1, df2])

    df = pd.concat([df,
                    get_df(accs_pdy, -1, 'score_de1'),
                    get_df(accs_palldy, -1, 'score_da'),
                    get_df(accs_pxy, -1, 'score_dx'),
                    ])
    return df


### Analysis of full examples

def get_dat_matched_idx(dat, eps=0.1, col_name='score', n_smp=None, **kwargs):

    matched = dat.apply(lambda s: get_balanced_index(*get_probs(s), eps=eps, **kwargs)
                        , axis=1)
    return matched


def get_balanced_index(p_xd, p_dy, p_xy, eps=0.01, ord=1, normalization=True,
                       direct_match=False, use_cooccur=False, temp_filter=False,
                       pxa_all_norm=False,
                       **kwargs):

    if direct_match:
        matched_idx_arr = return_match_idx(p_xd[:, :, 0], ord=ord, eps=eps, return_arr=True)
    else:
        matched_idx_arr = adj(p_xd, eps=eps, ord=ord, temp_filter=False, **kwargs)

    return matched_idx_arr


def gen_eg(ds, eps, spacy_model, file_pref=None):
    # eg: df[df['index']==98].iloc[0:1]
    matched_idx, norm_arr = get_dat_matched_idx(ds, eps=eps).iloc[0]
    cov, premise, interv, outcome, p_dy = ds.iloc[0][['covariates', 'text', 'interventions', 'outcome', 'p_dy']]




    X = [tt_ize(crop_sent(x.replace('$', r'\$'), spacy_model=spacy_model, sent_idx=0, offset=0)) for xidx, x in
         enumerate(cov)]
    X = [small_ize(rf"$\sfX_{{{xidx + 1}}}$: " + x) for xidx, x in enumerate(X) if len(x) > 0]

    E1 = small_ize(rf"$\sfE_1$: " + tt_ize(premise))
    E2 = small_ize(rf"$\sfE_2$: " + tt_ize(outcome))
    D = [small_ize(rf"$\sfA_{{{didx + 1}}}$: " + tt_ize(d)) for didx, d in enumerate(interv)]
    p_dy = [rf"${p[0]:.4f}$" for p in p_dy]
    norm_arr = [rf"$0$"] + [rf"${delta:.4f}$" for delta in norm_arr]
    #     print(matched_idx, len(D), len(X))

    for mi in matched_idx:
        D[mi] = highlight(D[mi])

    print(f"matched: {len(matched_idx)}, len(X): {len(X)}, len(D): {len(D)}")

    Xstr = "\n".join([x + r" & & & \\" for x in X])
    Dstr = "\n".join([r'\, \\'] + [d + r" \\" for d in [E1] + D])
    p_dy_str = "\n".join([r'\, \\'] + [p + r" \\" for p in p_dy])
    arr_str = "\n".join([r'\, \\'] + [delta + r" \\" for delta in norm_arr])
    if file_pref is not None:
        with open(f"{file_pref}_cov.txt", "w") as f:
            f.write(Xstr)
        with open(f"{file_pref}_d.txt", "w") as f:
            f.write(Dstr)
        with open(f"{file_pref}_pdy.txt", "w") as f:
            f.write(p_dy_str)
        with open(f"{file_pref}_dist.txt", "w") as f:
            f.write(arr_str)
    return Xstr, Dstr, E2, p_dy_str


### full ablations
def ablation_query(dt, D, F, S, Q, C, E):
    return dt.query(
        f"(normalization == {S}) and (direct_match=={D}) and (use_cooccur=={C}) and (temp_filter=={F}) and (res_norm=={E}) and (pxa_norm=={Q})")


def gen_config_str(res):
    d, f, s, q, c, e = res
    sf = lambda s: r"\textbf{" + s + r"}"
    on = lambda s, lb: sf(lb) if s else ""

    cfg_str = f'{on(d, "D")}{on(f, "F")}{on(s and not d, "S")}{on(q and not d, "Q")}{on(c, "C")}{on(e and not c, "E")}'
    if len(cfg_str) == 0:
        cfg_str = r"$\emptyset$"
    else:
        cfg_str = "+" + cfg_str
    return rf" {cfg_str} "


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
            min_idx, max_idx = np.where(res == res_min)[0], np.where(res == res_max)[0]
            #         print(res, res_min, res_max, min_idx, max_idx)
            res_str = [rf'${r:.3f}$' for r in res]
            best = res_str[max_idx[0]]
            worst = res_str[min_idx[0]]
            for idx in min_idx:
                res_str[idx] = r"\abpoor{" + res_str[idx] + r"}"
            for idx in max_idx:
                res_str[idx] = r"\abgood{" + res_str[idx] + r"}"
            res_str = [best, worst] + res_str
            print(f"{dataset}/{score}: {'&'.join(rf' {r} ' for r in res_str)}\n")


def gen_full_ablation_config():
    return [
        # D      F      S       Q     C      E
        (False, False, False, False, False, False),  ## 0
        # 1
        (True, False, True, True, False, False),  # D
        (False, True, False, False, False, False),  # F
        (False, False, True, False, False, False),  # S
        (False, False, False, True, False, False),  # Q
        (False, False, False, False, True, True),  # C
        (False, False, False, False, False, True),  # E

        # 2
        ## D: exclusive w/ S and Q
        ## set to T T T
        ## C: exclusive w/ E
        ## set to T T
        ## D
        (True, True, True, True, False, False),  ## DF
        (True, False, True, True, True, True),  ## DC
        (True, False, True, True, False, True),  ## DE

        ## F
        (False, True, True, False, False, False),  ## FS
        (False, True, False, True, False, False),  ## FQ
        (False, True, False, False, True, True),  ## FC
        (False, True, False, False, False, True),  # FE

        ## S
        (False, False, True, True, False, False),  ## SQ
        (False, False, True, False, True, True),  ## SC
        (False, False, True, False, False, True),  ## SE

        ## Q
        (False, False, False, True, True, True),  ## QC
        (False, False, False, True, False, True),  ## QE

        ## C - done

        ## 3

        ## DF
        (True, True, True, True, True, True),  ## DFC
        (True, True, True, True, False, True),  ## DFE

        ## DC - done
        ## DE - done
        ##  D F S Q C E
        ## FS
        (False, True, True, True, False, False),  ## FSQ
        (False, True, True, True, True, True),  ## FSC
        (False, True, True, True, False, True),  ## FSE

        ## FQ
        (False, True, False, True, True, True),  ## FQC
        (False, True, False, True, False, True),  ## FQE

        ## SQ
        (False, False, True, True, True, True),  ## SQC
        (False, False, True, True, False, True),  ## SQE

        ## 4

        (False, True, True, True, True, True),  ## FSQC
        (False, True, True, True, False, True),  ## FSQE
    ]