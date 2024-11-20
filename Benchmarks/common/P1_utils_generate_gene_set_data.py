def generate_gene_set_data(data, genes, gene_name_type='entrez',
    gene_set_category='c6.all', metric='mean', standardize=False, data_dir=
    '../../Data/examples/Gene_Sets/MSigDB.v7.0/'):
    """
    This function generates genomic data summarized at the gene set level.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].
    genes: 1-D array or list of gene names with a length of n_features. It indicates which gene a genomic
        feature belongs to.
    gene_name_type: string, indicating the type of gene name used in genes. 'entrez' indicates Entrez gene ID and
        'symbols' indicates HGNC gene symbol. Default is 'symbols'.
    gene_set_category: string, indicating the gene sets for which data will be calculated. 'c2.cgp' indicates gene sets
        affected by chemical and genetic perturbations; 'c2.cp.biocarta' indicates BioCarta gene sets; 'c2.cp.kegg'
        indicates KEGG gene sets; 'c2.cp.pid' indicates PID gene sets; 'c2.cp.reactome' indicates Reactome gene sets;
        'c5.bp' indicates GO biological processes; 'c5.cc' indicates GO cellular components; 'c5.mf' indicates
        GO molecular functions; 'c6.all' indicates oncogenic signatures. Default is 'c6.all'.
    metric: string, indicating the way to calculate gene-set-level data. 'mean' calculates the mean of gene
        features belonging to the same gene set. 'sum' calculates the summation of gene features belonging
        to the same gene set. 'max' calculates the maximum of gene features. 'min' calculates the minimum
        of gene features. 'abs_mean' calculates the mean of absolute values. 'abs_maximum' calculates
        the maximum of absolute values. Default is 'mean'.
    standardize: boolean, indicating whether to standardize features before calculation. Standardization transforms
        each feature to have a zero mean and a unit standard deviation.

    Returns:
    --------
    gene_set_data: a data frame of calculated gene-set-level data. Column names are the gene set names.
    """
    sample_name = None
    if isinstance(data, pd.DataFrame):
        sample_name = data.index
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    genes = [str(i) for i in genes]
    if gene_name_type == 'entrez':
        gene_set_category = gene_set_category + '.v7.0.entrez.gmt'
    if gene_name_type == 'symbols':
        gene_set_category = gene_set_category + '.v7.0.symbols.gmt'
    f = open(data_dir + gene_set_category, 'r')
    x = f.readlines()
    gene_sets = {}
    for i in range(len(x)):
        temp = x[i].split('\n')[0].split('\t')
        gene_sets[temp[0]] = temp[2:]
    gene_set_data = np.empty((data.shape[0], len(gene_sets)))
    gene_set_data.fill(np.nan)
    gene_set_names = np.array(list(gene_sets.keys()))
    for i in range(len(gene_set_names)):
        idi = np.where(np.isin(genes, gene_sets[gene_set_names[i]]))[0]
        if len(idi) > 0:
            if metric == 'sum':
                gene_set_data[:, i] = np.nansum(data[:, idi], axis=1)
            elif metric == 'max':
                gene_set_data[:, i] = np.nanmax(data[:, idi], axis=1)
            elif metric == 'min':
                gene_set_data[:, i] = np.nanmin(data[:, idi], axis=1)
            elif metric == 'abs_mean':
                gene_set_data[:, i] = np.nanmean(np.absolute(data[:, idi]),
                    axis=1)
            elif metric == 'abs_maximum':
                gene_set_data[:, i] = np.nanmax(np.absolute(data[:, idi]),
                    axis=1)
            else:
                gene_set_data[:, i] = np.nanmean(data[:, idi], axis=1)
    if sample_name is None:
        gene_set_data = pd.DataFrame(gene_set_data, columns=gene_set_names)
    else:
        gene_set_data = pd.DataFrame(gene_set_data, columns=gene_set_names,
            index=sample_name)
    keep_id = np.where(np.sum(np.invert(pd.isna(gene_set_data)), axis=0) > 0)[0
        ]
    gene_set_data = gene_set_data.iloc[:, keep_id]
    return gene_set_data
