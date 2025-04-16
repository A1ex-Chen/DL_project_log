def load_data_cv(self, fold):
    train_index = self.cv_train_indexes[fold]
    val_index = self.cv_val_indexes[fold]
    return self.load_data_by_index(train_index, val_index)
