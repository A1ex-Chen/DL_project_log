@staticmethod
def collate_fn(batch):
    img, label, path, shapes, masks = zip(*batch)
    batched_masks = torch.cat(masks, 0)
    for i, l in enumerate(label):
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0
        ), path, shapes, batched_masks
