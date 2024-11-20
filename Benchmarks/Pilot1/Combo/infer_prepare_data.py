def prepare_data(sample_set='NCI60', drug_set='ALMANAC', use_landmark_genes
    =False, preprocess_rnaseq=None):
    df_expr = NCI60.load_sample_rnaseq(use_landmark_genes=
        use_landmark_genes, preprocess_rnaseq=preprocess_rnaseq, sample_set
        =sample_set)
    df_desc = NCI60.load_drug_set_descriptors(drug_set=drug_set)
    return df_expr, df_desc
