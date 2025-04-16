def pre_train_config(self):
    print('Data sizes:\nTrain:')
    print('Data set: ' + self.drug_resp_trn_loader.dataset.data_source +
        ' Size: ' + str(len(self.drug_resp_trn_loader.dataset)))
    print('Validation:')
    self.val_index = 0
    for idx, loader in enumerate(self.drug_resp_val_loaders):
        print('Data set: ' + loader.dataset.data_source + ' Size: ' + str(
            len(loader.dataset)))
        if loader.dataset.data_source == self.args.train_sources:
            self.val_index = idx
