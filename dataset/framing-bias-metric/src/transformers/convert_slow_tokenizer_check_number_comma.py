def check_number_comma(piece: str) ->bool:
    return len(piece) < 2 or piece[-1] != ',' or not piece[-2].isdigit()
