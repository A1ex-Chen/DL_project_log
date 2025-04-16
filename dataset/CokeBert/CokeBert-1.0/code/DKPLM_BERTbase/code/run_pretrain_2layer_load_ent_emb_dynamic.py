def load_ent_emb_dynamic():
    with open('code/knowledge_bert/load_data/e1_e2.pkl', 'rb') as f:
        ent_neighbor = pickle.load(f)
    with open('code/knowledge_bert/load_data/e1_r.pkl', 'rb') as f:
        ent_r = pickle.load(f)
    with open('code/knowledge_bert/load_data/e1_outORin.pkl', 'rb') as f:
        ent_outORin = pickle.load(f)
    return ent_neighbor, ent_r, ent_outORin
