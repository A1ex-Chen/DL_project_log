def __init__(self, algorithm: Algorithm, num_classes: int, preprocessor:
    Preprocessor, class_to_category_dict: Dict[int, str],
    category_to_class_dict: Dict[str, int]):
    super().__init__()
    self.algorithm = algorithm
    self.num_classes = num_classes
    self.preprocessor = preprocessor
    self.class_to_category_dict = class_to_category_dict
    self.category_to_class_dict = category_to_class_dict
