def show_col_shapes(df: pd.DataFrame):
    for key, val in recorder.__data__.items():
        print(f'[{key}]: {val.shape}')
