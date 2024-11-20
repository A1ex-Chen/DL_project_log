def __init__(self, root, partition, subsite=True, laterality=True, behavior
    =True, grade=True, transform=None, target_transform=None):
    self.root = root
    self.partition = partition
    self.transform = transform
    self.target_transform = target_transform
    self.subsite = subsite
    self.laterality = laterality
    self.behavior = behavior
    self.grade = grade
    if self.partition == 'train':
        data_file = self.training_data_file
        label_file = self.training_label_file
    elif self.partition == 'test':
        data_file = self.test_data_file
        label_file = self.test_label_file
    else:
        raise ValueError("Partition must either be 'train' or 'test'.")
    self.data = np.load(os.path.join(self.root, data_file))
    self.targets = self.get_targets(label_file)
