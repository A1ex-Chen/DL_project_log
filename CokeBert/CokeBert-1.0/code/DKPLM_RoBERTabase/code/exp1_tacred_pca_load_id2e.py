def load_id2e():
    id2q = defaultdict(dict)
    q2ent = defaultdict(dict)
    with open('../../data/kg_embed/entity2id.txt', 'r') as fin:
        id2q[0] = '0'
        for line in fin:
            line = line.strip().split()
            if len(line) == 1:
                continue
            id2q[int(line[1]) + 1] = line[0]
    with open('../../data/kg_embed/entity_map.txt', 'r', encoding='utf-8'
        ) as fin:
        for line in fin:
            line = line.strip().split('\t')
            if len(line) == 1:
                continue
            q2ent[line[1]] = line[0]
    return id2q, q2ent
