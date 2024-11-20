def test_copa_predict(ds, predictor, top_k=5):

    def relu(x):
        return np.maximum(0.0, x)
    premise, choice1, choice2, q, lb = ds[['premise', 'choice1', 'choice2',
        'question', 'label']]
    res = [predictor.get_temp(premise, choice1, top_k=top_k), predictor.
        get_temp(premise, choice2, top_k=top_k)]
    befores = [(r[0] - r[1]) for r in res]
    afters = [(r[1] - r[0]) for r in res]
    if q == 'effect':
        if afters[0] == afters[1]:
            print(f'tie at {premise}/after: {afters[0]}')
            return -1
        return np.argmax(afters)
    else:
        if befores[0] == befores[1]:
            print(f'tie at {premise}/before: {befores[0]}')
            return -1
        return np.argmax(befores)
