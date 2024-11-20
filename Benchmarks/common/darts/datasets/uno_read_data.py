def read_data(self, data_file, partition):
    """Read in the H5 data"""
    if partition == 'train':
        gene_data = 'x_train_0'
        drug_data = 'x_train_1'
    else:
        gene_data = 'x_val_0'
        drug_data = 'x_val_1'
    gene_data = torch.tensor(pd.read_hdf(data_file, gene_data).values)
    drug_data = torch.tensor(pd.read_hdf(data_file, drug_data).values)
    data = {'gene_data': gene_data, 'drug_data': drug_data}
    return data
