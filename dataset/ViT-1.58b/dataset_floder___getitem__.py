def __getitem__(self, index: int) ->Tuple[Any, Any]:
    """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
    while True:
        try:
            path, target = self.samples[index]
            img = Image.open(path)
            img = img.resize((256, 256), Image.BILINEAR)
            sample = img.convert('RGB')
            break
        except Exception as e:
            print(e)
            index = random.randint(0, len(self.samples) - 1)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return sample, target
