def viz_acc_n(df, plt, dataset, score, epss=None, y_title=None, x_label='',
    lw=3, ls='-', legend=None, ylim=None, ncol=3):
    if epss is None:
        epss = np.sort(df.eps.unique())
        epss = epss[epss >= 0]
    metrics = {'score_dl1': '$\\hat{\\Delta}_1$', 'score_dl2':
        '$\\hat{\\Delta}_2$', 'score_de1':
        '$\\hat{\\Delta}_{\\mathsf{E}_1}$', 'score_da':
        '$\\hat{\\Delta}_{\\mathcal{A}}$', 'score_dx':
        '$\\hat{\\Delta}_{\\mathcal{X}}$'}
    dt = df[(df.dataset_name == dataset) & (df.score == score)]
    for eps_id, eps in enumerate(epss):
        dtt = dt[dt.eps == eps]
        Ns = dtt.N.unique()
        grp = dtt.groupby('N').acc
        mus = grp.mean().values
        ci = np.array(grp.apply(lambda s: sms.DescrStatsW(s.values).
            tconfint_mean()).tolist())
        plt.errorbar(Ns, mus, yerr=np.abs(mus - ci.T), fmt='-o', ls=ls,
            label=f'${eps}$', c=f'C{eps_id}')
    plt.xlabel(f'$N$' + x_label, fontsize=20)
    if y_title is not None:
        plt.ylabel(y_title, fontsize=20)
    if ylim is not None:
        plt.ylim(ylim)
    plt.gca().tick_params(axis='both', which='major', labelsize=18)
    plt.yticks([0.5, 0.55, 0.6, 0.65, 0.7])
    plt.xticks([5, 20, 40, 60, 80, 100])
    if legend is not None:
        plt.legend(title='$\\epsilon$', fontsize=15, loc=legend, ncol=ncol
            ).get_title().set_fontsize(22)
    plt.tight_layout()
