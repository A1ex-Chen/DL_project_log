def find_parent(self, label):
    target_node = findall(self.class_structure, filter_=lambda node: node.
        LabelName.lower() in label)[0]
    while self.find_key(target_node.LabelName.lower()) is None:
        target_node = target_node.parent
    return self.find_key(target_node.LabelName.lower())
