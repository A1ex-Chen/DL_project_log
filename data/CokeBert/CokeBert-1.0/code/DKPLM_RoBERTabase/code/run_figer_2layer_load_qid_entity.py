def load_qid_entity():
    qid_entity = dict()
    with open('../../data/kg_embed/entity_map.txt') as fin:
        for line in fin:
            ent, qid = line.strip().split('\t')
            qid_entity[qid] = ent
    return qid_entity
