def GPT_Metric(base_data_path, answer_path, response_dir):
    dataset = json.load(open(base_data_path, 'rb'))
    answer = json.load(open(answer_path, 'rb'))
    model_name = parse_model_name(answer_path)[:-5]
    cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_base_dir = os.path.join(response_dir, model_name, cur_time)
    os.makedirs(save_base_dir, exist_ok=True)
    response_path = os.path.join(save_base_dir, 'sqa_gpt_metric.jsonl')
    res_path = os.path.join(save_base_dir, 'results.jsonl')
    ans_file = open(response_path, 'w')
    res_file = open(res_path, 'w')
    rd_seed = 0
    random.seed(rd_seed)
    sample_ids = random.sample(range(2017), 250)
    sample_ids.sort()
    print(f'Preparing queries ...')
    query_list = []
    for id in tqdm(sample_ids):
        query = generate_query(dataset[id], answer[id])
        query_list.append(query)
    print('GPT evaluating ...')
    response_list = []
    score_list = []
    res_list = []
    for ids, query in tqdm(zip(sample_ids, query_list)):
        messages = [{'role': 'system', 'content': SYS_VQA}]
        messages.append({'role': 'user', 'content': query})
        while True:
            try:
                response = openai.ChatCompletion.create(model='gpt-4',
                    messages=messages, n=6, max_tokens=512)
                response_list.append(response)
                break
            except:
                continue
        cur_text = []
        cur_score = []
        for ans in response['choices']:
            cur_text.append(ans['message']['content'])
            score = parse_score(ans['message']['content'])
            if score != None:
                cur_score.append(score)
        input_tokens = response['usage']['prompt_tokens']
        comp_tokens = response['usage']['completion_tokens']
        cost = input_tokens * 0.03 / 1000 + comp_tokens * 0.06 / 1000
        cur_dict = {'ids': ids, 'query': query, 'response': cur_text,
            'scores': cur_score, 'cost': cost, 'origin_answer': answer[ids]}
        res_list.append(cur_dict)
        ans_file.write(json.dumps(response) + '\n')
        res_file.write(json.dumps(cur_dict) + '\n')
        ans_file.flush()
        res_file.flush()
        time.sleep(2)
    with open(os.path.join(save_base_dir, 'samples_results.json'), 'w',
        encoding='utf-8') as f:
        f.write(json.dumps(res_list, indent=4))
    total_score, total_cost = 0, 0
    for res in res_list:
        tmp = 0
        for i in res['scores']:
            tmp += i
        tmp /= len(res['scores'])
        total_score += tmp
        total_cost += res['cost']
    total_score /= len(res_list)
    print(
        f'GPT Evaluation on {model_name} completed. Avg:{total_score}, Cost:{total_cost}'
        )
    overall_res = {'model name': model_name, 'Avg Score': total_score,
        'Cost': total_cost}
    with open(os.path.join(save_base_dir, 'results.json'), 'w', encoding=
        'utf-8') as f:
        f.write(json.dumps(overall_res, indent=4))
    return response_list, res_list
