def analyze(self, test_data, result_data, type='base'):
    output = OrderedDict()
    for idx in range(len(test_data)):
        result = result_data[idx]
        test = test_data[idx]
        assert str(test['id']) == str(result['id']
            ), f"test id is {test['id']}, result id is {result['id']}. id should be the same"
        occ = test['occ']
        occ_sim = test['occ_sim']
        gender = test['gender']
        if occ == 'Waitress' and gender == 'female' and type == 'base':
            occ = 'Waiter'
        elif occ == 'Waitress' and gender == 'male' and type == 'cf':
            occ = 'Waiter'
        if occ_sim == 'Waitress' and gender == 'female' and type == 'base':
            occ_sim = 'Waiter'
        elif occ_sim == 'Waitress' and gender == 'male' and type == 'cf':
            occ_sim = 'Waiter'
        key = f'{occ}+{occ_sim}+{gender}'
        if key not in output:
            output[key] = []
        sub_result = {'id': result['id'], 'metric_result': result[
            'metric_result'], 'ppl_results': result['ppl_results'], 'probs':
            result['probs'], 'gt_choice': int(result['gt_choice'])}
        output[key].append(sub_result)
    return output
