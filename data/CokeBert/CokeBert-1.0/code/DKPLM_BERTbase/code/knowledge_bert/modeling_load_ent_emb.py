def load_ent_emb():
    with open('code/knowledge_bert/load_data_test/e1_e2_list_2D_Tensor.pkl',
        'rb') as f:
        ent_neighbor = pickle.load(f)
    with open('code/knowledge_bert/load_data_test/e1_r_list_2D_Tensor.pkl',
        'rb') as f:
        ent_r = pickle.load(f)
    with open(
        'code/knowledge_bert/load_data_test/e1_outORin_list_2D_Tensor.pkl',
        'rb') as f:
        ent_outORin = pickle.load(f)
    return ent_neighbor, ent_r, ent_outORin
