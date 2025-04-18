def gather_cov_inter_data(cur_args):
    with open(cur_args.data_path) as fin:
        origin_data = [json.loads(data_line) for data_line in fin]
    with open(cur_args.cov_path) as fin:
        cov_data = [json.loads(data_line) for data_line in fin]
    with open(cur_args.inter_path) as fin:
        inter_data = [json.loads(data_line) for data_line in fin]
    if cur_args.debug:
        origin_data, cov_data, inter_data = origin_data[:1], cov_data[:1
            ], inter_data[:1]
    cov_inter_list = []
    for event_seq, cov_seq, inter_seq in zip(origin_data, cov_data, inter_data
        ):
        for i in range(0, 4):
            cov_event = cov_seq[f's{i}']
            inter_event = [event_seq['story'][i]] + inter_seq[f's{i}']
            cov_inter_pair = {'text': cov_event, 'outcome': inter_event}
            cov_inter_list.append(cov_inter_pair)
    print(len(cov_inter_list[0]['text']), len(cov_inter_list[0]['outcome']))
    return cov_inter_list
