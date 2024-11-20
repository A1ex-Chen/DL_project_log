def add_location_symbols(quantized_size, locate_special_token=0):
    custom_sp_symbols = []
    for symbol in SPECIAL_SYMBOLS:
        custom_sp_symbols.append(symbol)
    for symbol in [BOP_SYMBOL, EOP_SYMBOL, BOO_SYMBOL, EOO_SYMBOL, DOM_SYMBOL]:
        custom_sp_symbols.append(symbol)
    if locate_special_token > 0:
        custom_sp_symbols.append(GRD_SYMBOL)
    for i in range(quantized_size ** 2):
        token_name = f'<patch_index_{str(i).zfill(4)}>'
        custom_sp_symbols.append(token_name)
    return custom_sp_symbols
