def tfds_map(self, example):
    """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. 
        This method converts examples to the correct format."""
    if len(self.get_labels()) > 1:
        example.label = self.get_labels()[int(example.label)]
    return example
