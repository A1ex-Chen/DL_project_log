def preprocess(self, targets, batch_size, scale_tensor):
    targets_list = np.zeros((batch_size, 1, 5)).tolist()
    for i, item in enumerate(targets.cpu().numpy().tolist()):
        targets_list[int(item[0])].append(item[1:])
    max_len = max(len(l) for l in targets_list)
    targets = torch.from_numpy(np.array(list(map(lambda l: l + [[-1, 0, 0, 
        0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(targets.
        device)
    batch_target = targets[:, :, 1:5].mul_(scale_tensor)
    targets[..., 1:] = xywh2xyxy(batch_target)
    return targets
