def agg_generations(self, gen):
    agg = {}
    for lists in gen:
        for ctrl, sent in lists:
            if ctrl not in agg:
                agg[ctrl] = []
            agg[ctrl].append(sent)
    return agg
