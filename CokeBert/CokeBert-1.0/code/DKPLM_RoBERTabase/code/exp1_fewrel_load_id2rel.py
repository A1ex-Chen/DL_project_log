def load_id2rel():
    id2p = defaultdict(dict)
    p2rel = defaultdict(dict)
    with open('../../data/kg_embed/relation2id.txt', 'r') as fin:
        id2p[0] = '0'
        for line in fin:
            line = line.strip().split()
            if len(line) == 1:
                continue
            id2p[int(line[1]) + 1] = line[0]
    with open('../../data/kg_embed/pid2rel_all.json', 'r', encoding='utf-8'
        ) as fin_:
        fin = json.load(fin_)
        for pid, rel in fin.items():
            p2rel[pid] = rel[0]
    return id2p, p2rel
