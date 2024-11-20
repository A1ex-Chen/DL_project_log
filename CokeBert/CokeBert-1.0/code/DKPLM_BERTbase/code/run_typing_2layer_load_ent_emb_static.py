def load_ent_emb_static():
    with open('../../data/load_data_n/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
        ent_neighbor = pickle.load(f)
    with open('../../data/load_data_n/e1_r_list_2D_Tensor.pkl', 'rb') as f:
        ent_r = pickle.load(f)
    with open('../../data/load_data_n/e1_outORin_list_2D_Tensor.pkl', 'rb'
        ) as f:
        ent_outORin = pickle.load(f)
    return ent_neighbor, ent_r, ent_outORin
