def load_k_v_queryR_small(input_ent, candidate, ent_neighbor, ent_r,
    ent_outORin, embed_ent, embed_r, hop=2):
    assert hop in [1, 2], '"hop" should be chosen from [1, 2].'
    input_ent = input_ent.cpu()
    candidate = candidate.cpu()
    ent_pos_s = torch.nonzero(input_ent)
    max_entity = 0
    value = 0
    idx_1 = 0
    last_part = 0
    for idx_2, x in enumerate(ent_pos_s):
        if int(x[0]) != value:
            max_entity = max(idx_2 - idx_1, max_entity)
            idx_1 = idx_2
            value = int(x[0])
            last_part = 1
        else:
            last_part += 1
    max_entity = max(last_part, max_entity)
    new_input_ent = list()
    cand_pos_tensor = torch.LongTensor(input_ent.shape[0], candidate.shape[1])
    for i_th, ten in enumerate(input_ent):
        ten_ent = ten[ten != 0]
        ten_ent_uniqe = ten_ent.unique()
        cand_pos_tensor[i_th][:] = torch.LongTensor([int(ent in
            ten_ent_uniqe) for ent in candidate[0]])
        new_input_ent.append(torch.cat((ten_ent, torch.LongTensor([0] * (
            max_entity - ten_ent.shape[0])))))
    input_ent = torch.stack(new_input_ent)
    k_1, v_1, k_2, v_2 = k_v(input_ent, ent_neighbor, ent_r, ent_outORin,
        embed_ent, embed_r, hop)
    k_cand_1, v_cand_1, k_cand_2, v_cand_2 = k_v(candidate, ent_neighbor,
        ent_r, ent_outORin, embed_ent, embed_r, hop)
    return (k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2,
        cand_pos_tensor)
