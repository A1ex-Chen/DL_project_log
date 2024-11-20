def _check_exists(self):
    return os.path.exists(os.path.join(self.processed_folder, self.
        training_data_file)) and os.path.exists(os.path.join(self.
        processed_folder, self.training_label_file)) and os.path.exists(os.
        path.join(self.processed_folder, self.test_data_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.
        test_label_file))
