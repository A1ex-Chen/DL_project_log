@staticmethod
def collate_fn(batch):
    img, label, path, shapes = zip(*batch)
    for i, l in enumerate(label):
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes
