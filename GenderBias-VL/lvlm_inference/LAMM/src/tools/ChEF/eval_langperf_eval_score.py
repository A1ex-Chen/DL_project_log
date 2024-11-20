def eval_score(response_list):
    total = 0
    cnt = 0
    for item in response_list:
        text = item['choices'][0]['message']['content']
        score = parse_score(text)
        if score != None and score >= 0:
            total += score
            cnt += 1
    return total / cnt * 10.0
