def __len__(self):
    if self.mode == 'train':
        return 2048 * 16
    else:
        return len(self.models)
