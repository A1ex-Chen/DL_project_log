def gather_inter_outcome_data(cur_args):
    with open(cur_args.data_path) as fin:
        origin_data = [json.loads(data_line) for data_line in fin]
    with open(cur_args.inter_path) as fin:
        inter_data = [json.loads(data_line) for data_line in fin]
    if cur_args.debug:
        origin_data, inter_data = origin_data[:1], inter_data[:1]
    inter_outcome_list = []
    for event_seq, inter_seq in zip(origin_data, inter_data):
        for i in range(0, 4):
            inter_event = [event_seq['story'][i]] + inter_seq[f's{i}']
            outcome_event = [event_seq['story'][4]]
            inter_outcome_pair = {'text': inter_event, 'outcome': outcome_event
                }
            inter_outcome_list.append(inter_outcome_pair)
    print(len(inter_outcome_list[0]['text']), len(inter_outcome_list[0][
        'outcome']))
    return inter_outcome_list
