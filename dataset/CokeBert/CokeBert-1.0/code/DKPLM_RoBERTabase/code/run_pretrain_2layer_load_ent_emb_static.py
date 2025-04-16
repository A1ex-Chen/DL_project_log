def load_ent_emb_static():
    with open('../../data/load_data_n/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
        ent_neighbor = pickle.load(f)
    print('ent_neighbor:', ent_neighbor.shape)
    with open('../../data/load_data_n/e1_r_list_2D_Tensor.pkl', 'rb') as f:
        ent_r = pickle.load(f)
    print('ent_r:', ent_r.shape)
    with open('../../data/load_data_n/e1_outORin_list_2D_Tensor.pkl', 'rb'
        ) as f:
        ent_outORin = pickle.load(f)
    print('ent_outORin:', ent_outORin.shape)
    return ent_neighbor, ent_r, ent_outORin
