def find_columns_with_str(df, substr):
    col_indices = [df.columns.get_loc(col) for col in df.columns if substr in
        col]
    return col_indices
