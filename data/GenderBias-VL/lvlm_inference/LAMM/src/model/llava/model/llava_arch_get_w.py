def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if 
        keyword in k}
