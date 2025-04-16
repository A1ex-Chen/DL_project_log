@staticmethod
def collate_fn4(batch):
    im, label, path, shapes = zip(*batch)
    n = len(shapes) // 4
    im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]
    ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
    for i in range(n):
        i *= 4
        if random.random() < 0.5:
            im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=
                2.0, mode='bilinear', align_corners=False)[0].type(im[i].type()
                )
            lb = label[i]
        else:
            im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((
                im[i + 2], im[i + 3]), 1)), 2)
            lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo,
                label[i + 3] + ho + wo), 0) * s
        im4.append(im1)
        label4.append(lb)
    for i, lb in enumerate(label4):
        lb[:, 0] = i
    return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4
