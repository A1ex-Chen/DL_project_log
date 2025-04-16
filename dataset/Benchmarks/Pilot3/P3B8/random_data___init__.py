def __init__(self, doc_length=512, num_vocab=1000, num_docs=100, num_classes=10
    ):
    self.doc_length = doc_length
    self.num_vocab = num_vocab
    self.num_docs = num_docs
    self.num_classes = num_classes
    self.docs = self.create_docs(doc_length, num_vocab, num_docs)
    self.masks = self.create_masks(doc_length, num_docs)
    self.segment_ids = self.create_segment_ids(doc_length, num_docs)
    self.labels = self.create_labels(num_classes, num_docs)
