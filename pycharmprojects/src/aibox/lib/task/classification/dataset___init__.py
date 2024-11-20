def __init__(self, datasets: List[Dataset]):
    super().__init__(datasets)
    assert len(datasets) > 0
    dataset: Dataset = self.datasets[0]
    for i in range(1, len(datasets)):
        assert dataset.class_to_category_dict == datasets[i
            ].class_to_category_dict
        assert dataset.category_to_class_dict == datasets[i
            ].category_to_class_dict
        assert dataset.num_classes() == datasets[i].num_classes()
    self.master = dataset
