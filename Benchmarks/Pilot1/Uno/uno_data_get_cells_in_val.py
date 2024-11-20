def get_cells_in_val(self):
    val_cell_ids = list(set(self.df_response.loc[self.val_indexes[0]][
        'Sample'].values))
    return val_cell_ids
