def load_ent_emb_dynamic(data_dir):
    with open(os.path.join(data_dir, 'kg_neighbor/e1_e2.pkl'), 'rb') as f:
        ent_neighbor = pickle.load(f)
    with open(os.path.join(data_dir, 'kg_neighbor/e1_r.pkl'), 'rb') as f:
        ent_r = pickle.load(f)
    with open(os.path.join(data_dir, 'kg_neighbor/e1_outORin.pkl'), 'rb') as f:
        ent_outORin = pickle.load(f)
    return ent_neighbor, ent_r, ent_outORin
