def __getitem__(self, idx):
    return {'tokens': self.docs[idx], 'masks': self.masks[idx], 'seg_ids':
        self.segment_ids[idx], 'label': self.labels[idx]}
