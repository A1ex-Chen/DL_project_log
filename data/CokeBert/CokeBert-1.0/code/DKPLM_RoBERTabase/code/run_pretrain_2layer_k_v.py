def k_v(input_ent):
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
        shape[1], 100)
    input_ent_r_2 = torch.index_select(ent_r, 0, input_ent_neighbor.reshape
        (input_ent_neighbor.shape[0] * input_ent_neighbor.shape[1])).long()
    input_ent_r_emb_2 = torch.index_select(embed_r, 0, input_ent_r_2.
        reshape(input_ent_r_2.shape[0] * input_ent_r_2.shape[1]))
    input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],
        input_ent.shape[1], ent_r.shape[1], ent_neighbor.shape[1], 100)
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
