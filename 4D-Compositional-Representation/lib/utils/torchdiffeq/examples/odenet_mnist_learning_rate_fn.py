def learning_rate_fn(itr):
    lt = [(itr < b) for b in boundaries] + [True]
    i = np.argmax(lt)
    return vals[i]
