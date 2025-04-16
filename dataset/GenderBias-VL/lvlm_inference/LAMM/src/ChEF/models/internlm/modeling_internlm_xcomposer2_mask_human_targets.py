def mask_human_targets(self, input_ids, pure=False):
    target_batch = []
    for bs in range(input_ids.shape[0]):
        ids = input_ids[bs]
        targets = copy.deepcopy(ids)
        end_count = 0
        last_eoa = 0
        for i, temp_id in enumerate(ids):
            if temp_id == 92542:
                if end_count % 2 == 0:
                    targets[last_eoa:i + 6] = -100
                else:
                    last_eoa = i + 1
                end_count += 1
            elif temp_id == 2:
                targets[i + 1:] = -100
                break
        if temp_id != 2 and end_count % 2 == 0:
            targets[last_eoa + 1:] = -100
        target_batch.append(targets.unsqueeze(0))
    target_batch = torch.cat(target_batch, dim=0)
    return target_batch
