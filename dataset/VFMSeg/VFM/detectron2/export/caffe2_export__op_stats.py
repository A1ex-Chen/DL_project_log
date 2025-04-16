def _op_stats(net_def):
    type_count = {}
    for t in [op.type for op in net_def.op]:
        type_count[t] = type_count.get(t, 0) + 1
    type_count_list = sorted(type_count.items(), key=lambda kv: kv[0])
    type_count_list = sorted(type_count_list, key=lambda kv: -kv[1])
    return '\n'.join('{:>4}x {}'.format(count, name) for name, count in
        type_count_list)
