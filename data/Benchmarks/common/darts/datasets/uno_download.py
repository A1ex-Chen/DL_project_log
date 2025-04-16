def download(self):
    """Download the Synthetic data if it doesn't exist in processed_folder already."""
    if self._check_exists():
        return
    makedir_exist_ok(self.raw_folder)
    makedir_exist_ok(self.processed_folder)
    for url in self.urls:
        filename = url.rpartition('/')[2]
        file_path = os.path.join(self.raw_folder, filename)
        download_url(url, root=self.raw_folder, filename=filename, md5=None)
    print('Processing...')
    training_set = self.read_data(os.path.join(self.raw_folder,
        'top_21_auc_1fold.uno.h5'), 'train'), self.read_targets(os.path.
        join(self.raw_folder, 'top_21_auc_1fold.uno.h5'), 'train')
    test_set = self.read_data(os.path.join(self.raw_folder,
        'top_21_auc_1fold.uno.h5'), 'test'), self.read_targets(os.path.join
        (self.raw_folder, 'top_21_auc_1fold.uno.h5'), 'test')
    train_data_path = os.path.join(self.processed_folder, self.
        training_data_file)
    torch.save(training_set[0], train_data_path)
    train_label_path = os.path.join(self.processed_folder, self.
        training_label_file)
    torch.save(training_set[1], train_label_path)
    test_data_path = os.path.join(self.processed_folder, self.test_data_file)
    torch.save(test_set[0], test_data_path)
    test_label_path = os.path.join(self.processed_folder, self.test_label_file)
    torch.save(test_set[1], test_label_path)
    print('Done!')
