def load_batch_k_v_queryR(input_ent, max_neighbor=4):
    input_ent = input_ent.cpu()
    input_ent_neighbor_emb = torch.zeros(input_ent.shape[0], input_ent.
        shape[1], max_neighbor).long()
    input_ent_r_emb = torch.zeros(input_ent.shape[0], input_ent.shape[1],
        max_neighbor).long()
    input_ent_outORin_emb = torch.zeros(input_ent.shape[0], input_ent.shape
        [1], max_neighbor)
    ent_pos_s = torch.nonzero(input_ent)
    ents = input_ent[input_ent != 0]
    for i, ent in enumerate(ents):
        neighbor_length = len(ent_neighbor[int(ent)])
        if neighbor_length < max_neighbor:
            input_ent_neighbor_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][
                :] = torch.LongTensor(ent_neighbor[int(ent)] + [0] * (
                max_neighbor - neighbor_length))
        else:
            input_ent_neighbor_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][
                :] = torch.LongTensor(ent_neighbor[int(ent)][:max_neighbor])
        r_length = len(ent_r[int(ent)])
        if r_length < max_neighbor:
            input_ent_r_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:
                ] = torch.LongTensor(ent_r[int(ent)] + [0] * (max_neighbor -
                r_length))
        else:
            input_ent_r_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:
                ] = torch.LongTensor(ent_r[int(ent)][:max_neighbor])
        outORin_length = len(ent_outORin[int(ent)])
        if outORin_length < max_neighbor:
            input_ent_outORin_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:
                ] = torch.FloatTensor(ent_outORin[int(ent)] + [0] * (
                max_neighbor - outORin_length))
        else:
            input_ent_outORin_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:
                ] = torch.FloatTensor(ent_outORin[int(ent)][:max_neighbor])
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(
        input_ent_neighbor_emb.shape[0] * input_ent_neighbor_emb.shape[1],
        max_neighbor)
    input_ent_neighbor_emb = torch.index_select(embed_ent, 0,
        input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0] *
        input_ent_neighbor_emb.shape[1]))
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent.shape
        [0], input_ent.shape[1], max_neighbor, 100)
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent_r_emb.shape[0] *
        input_ent_r_emb.shape[1], max_neighbor)
    input_ent_r_emb = torch.index_select(embed_ent, 0, input_ent_r_emb.
        reshape(input_ent_r_emb.shape[0] * input_ent_r_emb.shape[1]))
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent.shape[0], input_ent
        .shape[1], max_neighbor, 100)
    input_ent_outORin_emb = input_ent_outORin_emb.unsqueeze(3)
    k = input_ent_outORin_emb.cuda() * input_ent_r_emb.cuda()
    v = input_ent_neighbor_emb.cuda() + k
    return k, v
