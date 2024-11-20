def load_k_v_queryR_small(input_ent, ent_neighbor, ent_r, ent_outORin,
    embed_ent, embed_r):
    input_ent = input_ent.cpu()
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
    for i_th, ten in enumerate(input_ent):
        ten_ent = ten[ten != 0]
        new_input_ent.append(torch.cat((ten_ent, torch.LongTensor([0] * (
            max_entity - ten_ent.shape[0])))))
    input_ent = torch.stack(new_input_ent)
    input_ent_neighbor = torch.index_select(ent_neighbor, 0, input_ent.
        reshape(input_ent.shape[0] * input_ent.shape[1])).long()
    input_ent_neighbor_emb_1 = torch.index_select(embed_ent, 0,
        input_ent_neighbor.reshape(input_ent_neighbor.shape[0] *
        input_ent_neighbor.shape[1]))
    input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.
        shape[0], input_ent.shape[1], ent_neighbor.shape[1], embed_ent.
        shape[-1])
    input_ent_r_emb_1 = torch.index_select(ent_r, 0, input_ent.reshape(
        input_ent.shape[0] * input_ent.shape[1])).long()
    input_ent_r_emb_1 = torch.index_select(embed_r, 0, input_ent_r_emb_1.
        reshape(input_ent_r_emb_1.shape[0] * input_ent_r_emb_1.shape[1]))
    input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],
        input_ent.shape[1], ent_r.shape[1], embed_r.shape[-1])
    input_ent_outORin_emb_1 = torch.index_select(ent_outORin, 0, input_ent.
        reshape(input_ent.shape[0] * input_ent.shape[1]))
    input_ent_outORin_emb_1 = input_ent_outORin_emb_1.reshape(input_ent.
        shape[0], input_ent.shape[1], input_ent_outORin_emb_1.shape[1])
    input_ent_outORin_emb_1 = input_ent_outORin_emb_1.unsqueeze(3)
    input_ent_neighbor_2 = torch.index_select(ent_neighbor, 0,
        input_ent_neighbor.reshape(input_ent_neighbor.shape[0] *
        input_ent_neighbor.shape[1])).long()
    input_ent_neighbor_emb_2 = torch.index_select(embed_ent, 0,
        input_ent_neighbor_2.reshape(input_ent_neighbor_2.shape[0] *
        input_ent_neighbor_2.shape[1]))
    input_ent_neighbor_emb_2 = input_ent_neighbor_emb_2.reshape(input_ent.
        shape[0], input_ent.shape[1], ent_neighbor.shape[1], ent_neighbor.
        shape[1], embed_ent.shape[-1])
    input_ent_r_2 = torch.index_select(ent_r, 0, input_ent_neighbor.reshape
        (input_ent_neighbor.shape[0] * input_ent_neighbor.shape[1])).long()
    input_ent_r_emb_2 = torch.index_select(embed_r, 0, input_ent_r_2.
        reshape(input_ent_r_2.shape[0] * input_ent_r_2.shape[1]))
    input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],
        input_ent.shape[1], ent_r.shape[1], ent_neighbor.shape[1], embed_r.
        shape[-1])
    input_ent_outORin_emb_2 = torch.index_select(ent_outORin, 0,
        input_ent_neighbor.reshape(input_ent_neighbor.shape[0] *
        input_ent_neighbor.shape[1]))
    input_ent_outORin_emb_2 = input_ent_outORin_emb_2.reshape(input_ent_r_emb_2
        .shape[0], input_ent_r_emb_2.shape[1], input_ent_r_emb_2.shape[2],
        input_ent_r_emb_2.shape[3])
    input_ent_outORin_emb_2 = input_ent_outORin_emb_2.unsqueeze(4)
    k_1 = input_ent_outORin_emb_1.cuda() * input_ent_r_emb_1.cuda()
    v_1 = input_ent_neighbor_emb_1.cuda() + k_1
    k_2 = input_ent_outORin_emb_2.cuda() * input_ent_r_emb_2.cuda()
    v_2 = input_ent_neighbor_emb_2.cuda() + k_2
    return k_1, v_1, k_2, v_2
