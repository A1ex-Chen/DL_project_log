def generate_mixed_data_with_SAM(src_sam_masks_data, trg_sam_masks_data,
    src_img_indices, src_2d_data, src_3d_data, src_seg_labels,
    trg_img_indices, trg_2d_data, trg_3d_data, trg_2d_pl, trg_3d_pl, device=0):
    src_2d_with_trg_mix = []
    trg_2d_with_src_mix = []
    src_2d_with_trg_mix_label = []
    trg_2d_with_src_mix_label = []
    src_3d_with_trg_mix = []
    trg_3d_with_src_mix = []
    mixed_batch = {}
    random_mix = {}
    cut_mix = {}
    end = time.time()
    ids_lists = [sample(list(discard(set(src_masks.numpy().flatten()))),
        list(discard(set(trg_masks.numpy().flatten())))) for src_masks,
        trg_masks in zip(src_sam_masks_data, trg_sam_masks_data)]
    data_time1 = time.time() - end
    end = time.time()
    for batch_id in range(len(src_sam_masks_data)):
        src_mask = (src_sam_masks_data[batch_id] == ids_lists[batch_id][0][0]
            ).numpy()
        for id in range(1, len(ids_lists[batch_id][0])):
            src_mask = src_mask | (src_sam_masks_data[batch_id] ==
                ids_lists[batch_id][0][id]).numpy()
        trg_mask = (trg_sam_masks_data[batch_id] == ids_lists[batch_id][1][0]
            ).numpy()
        for id in range(1, len(ids_lists[batch_id][1])):
            trg_mask = trg_mask | (trg_sam_masks_data[batch_id] ==
                ids_lists[batch_id][1][id]).numpy()
        src_img_indices
        src_2d_with_trg_mix_label
        trg_2d_with_src_mix_label
        src_mask = torch.from_numpy(src_mask).unsqueeze(0).repeat(3, 1, 1)
        trg_mask = torch.from_numpy(trg_mask).unsqueeze(0).repeat(3, 1, 1)
        src_mix = src_2d_data[batch_id].clone()
        src_mix[trg_mask] = trg_2d_data[batch_id][trg_mask]
        src_2d_with_trg_mix.append(src_mix)
        trg_mix = trg_2d_data[batch_id].clone()
        trg_mix[src_mask] = src_2d_data[batch_id][src_mask]
        trg_2d_with_src_mix.append(trg_mix)
    data_time3 = time.time() - end
    return mixed_batch, random_mix, cut_mix
