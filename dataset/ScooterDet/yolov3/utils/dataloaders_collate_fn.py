@staticmethod
def collate_fn(batch):
    im, label, path, shapes = zip(*batch)
    for i, lb in enumerate(label):
        lb[:, 0] = i
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes
