def map_val_table_path(self):
    self.val_table_map = {}
    print('Mapping dataset')
    for i, data in enumerate(tqdm(self.val_table.data)):
        self.val_table_map[data[3]] = data[0]
