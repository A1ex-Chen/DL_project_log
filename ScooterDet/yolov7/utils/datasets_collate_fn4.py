@staticmethod
def collate_fn4(batch):
    img, label, path, shapes = zip(*batch)
    n = len(shapes) // 4
    img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]
    ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
    for i in range(n):
        i *= 4
        if random.random() < 0.5:
            im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=
                2.0, mode='bilinear', align_corners=False)[0].type(img[i].
                type())
            l = label[i]
        else:
            im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((
                img[i + 2], img[i + 3]), 1)), 2)
            l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, 
                label[i + 3] + ho + wo), 0) * s
        img4.append(im)
        label4.append(l)
    for i, l in enumerate(label4):
        l[:, 0] = i
    return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4
