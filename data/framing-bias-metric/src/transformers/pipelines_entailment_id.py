@property
def entailment_id(self):
    for label, ind in self.model.config.label2id.items():
        if label.lower().startswith('entail'):
            return ind
    return -1
