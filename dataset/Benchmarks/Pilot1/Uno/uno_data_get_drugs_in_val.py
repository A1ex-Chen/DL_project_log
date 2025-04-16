def get_drugs_in_val(self):
    if np.isin('Drug', self.df_response.columns.values):
        val_drug_ids = list(set(self.df_response.loc[self.val_indexes[0]][
            'Drug'].values))
    else:
        val_drug_ids = list(set(self.df_response.loc[self.val_indexes[0]][
            'Drug1'].values))
    return val_drug_ids
