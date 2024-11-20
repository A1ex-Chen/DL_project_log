def __init__(self, tokenizer, datasets_paths):
    self.tokenizer = tokenizer
    self.datasets_paths = datasets_paths,
    self.datasets = [load_dataset(dataset_path) for dataset_path in self.
        datasets_paths[0]]
    self.train_data = concatenate_datasets([dataset['train'] for dataset in
        self.datasets])
    self.test_data = concatenate_datasets([dataset['test'] for dataset in
        self.datasets])
    self.image_normalize = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
