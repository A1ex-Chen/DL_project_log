def load_k_v_queryR(input_ent):
    input_ent = input_ent.cpu()
    input_ent_neighbor_emb = torch.index_select(ent_neighbor, 0, input_ent.
        reshape(input_ent.shape[0] * input_ent.shape[1])).long()
    input_ent_neighbor_emb = torch.index_select(embed_ent, 0,
        input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0] *
        input_ent_neighbor_emb.shape[1]))
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent.shape
        [0], input_ent.shape[1], ent_neighbor.shape[1], 100)
    input_ent_r_emb = torch.index_select(ent_r, 0, input_ent.reshape(
        input_ent.shape[0] * input_ent.shape[1])).long()
    input_ent_r_emb = torch.index_select(embed_ent, 0, input_ent_r_emb.
        reshape(input_ent_r_emb.shape[0] * input_ent_r_emb.shape[1]))
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent.shape[0], input_ent
        .shape[1], ent_r.shape[1], 100)
    input_ent_outORin_emb = torch.index_select(ent_outORin, 0, input_ent.
        reshape(input_ent.shape[0] * input_ent.shape[1]))
    input_ent_outORin_emb = input_ent_outORin_emb.reshape(input_ent.shape[0
        ], input_ent.shape[1], input_ent_outORin_emb.shape[1])
    input_ent_outORin_emb = input_ent_outORin_emb.unsqueeze(3)
    k = input_ent_outORin_emb.cuda() * input_ent_r_emb.cuda()
    v = input_ent_neighbor_emb.cuda() + k
    return k, v
