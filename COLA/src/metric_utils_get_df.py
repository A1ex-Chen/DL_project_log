def get_df(accs, eps, score_lb):
    df = pd.DataFrame(accs, columns=['acc'])
    df['eps'] = eps
    df['score'] = score_lb
    return df
