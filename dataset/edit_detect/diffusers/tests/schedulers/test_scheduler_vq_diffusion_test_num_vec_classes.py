def test_num_vec_classes(self):
    for num_vec_classes in [5, 100, 1000, 4000]:
        self.check_over_configs(num_vec_classes=num_vec_classes)
