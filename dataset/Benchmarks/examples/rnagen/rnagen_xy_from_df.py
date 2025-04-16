def xy_from_df(df, shuffle=False):
    if shuffle:
        df = df.sample(frac=1, random_state=0)
    x = df.iloc[:, 2:].values
    y = pd.get_dummies(df.type).values
    return x, y
