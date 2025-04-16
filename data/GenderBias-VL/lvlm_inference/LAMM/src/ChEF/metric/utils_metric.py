def metric(self, answer_path):
    with open(answer_path, 'r', encoding='utf8') as f:
        answers = json.load(f)
    results, metric_answers = self.metric_func(answers)
    with open(answer_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(metric_answers, indent=4, ensure_ascii=False))
    if isinstance(results, dict):
        print(f'{self.dataset_name}:')
        for key, value in results.items():
            print(f'{key}: {value}')
        return results
    else:
        print(f'{self.dataset_name}: {results}')
        return results
